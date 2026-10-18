#!/usr/bin/env python3
"""Regression tests for RNodeInterface modem selection (`modem = lora|gfsk|fsk`).

These run without any hardware. The interface is constructed with a fake
serial port that records the exact bytes it is asked to write, so that:

  * the LoRa byte stream can be asserted to be unchanged by the modem work
    (the single most important compatibility property), and
  * the (G)FSK/FSK paths can be asserted to add exactly one CMD_MODEM frame,
    in the right position, with the right payload, and
  * the opt-in statistics polling can be asserted to be entirely absent from
    the default byte stream, and to emit the CMD_STAT_RX/CMD_STAT_TX request
    frames only when `poll_stats` is enabled.

Usage:

    python3 tests/test_rnode_modem.py

Exits non-zero if any check fails.
"""

import os
import pty
import sys
import threading
import time

import serial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import RNS
from RNS.Interfaces.RNodeInterface import RNodeInterface, KISS
from RNS.vendor.configobj import ConfigObj


class _StubReticulum:
    """Stand-in for a running Reticulum instance.

    Interface.__init__() reads eleven ingress/egress-control defaults from the
    live Reticulum instance. These tests deliberately do not start one (there
    is no config, no instance and no radio), so the accessor is patched to
    return the class defaults from Interface. The values only populate
    ingress-control fields and are irrelevant to what is under test.
    """

    def _default_ic_max_held_announces(self):    return 256
    def _default_ic_burst_hold(self):            return 15
    def _default_ic_burst_freq_new(self):        return 3
    def _default_ic_burst_freq(self):            return 10
    def _default_ic_pr_burst_freq_new(self):     return 3
    def _default_ic_pr_burst_freq(self):         return 8
    def _default_ic_new_time(self):              return 7200
    def _default_ic_burst_penalty(self):         return 15
    def _default_ic_held_release_interval(self): return 5
    def _default_ec_pr_freq(self):               return 5
    def _default_egress_control(self):           return False


RNS.Reticulum.get_instance = staticmethod(lambda: _StubReticulum())


class FakeSerial:
    """Records everything written to it, and decodes it back into frames."""

    def __init__(self):
        self.writes = []
        self.is_open = True

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def close(self):
        self.is_open = False

    def frames(self):
        out = []
        for w in self.writes:
            assert w[0] == KISS.FEND and w[-1] == KISS.FEND, "malformed frame %r" % (w,)
            body = w[1:-1]
            if not body:
                continue
            out.append((body[0], body[1:]))
        return out

    def commands(self):
        return [c for c, _ in self.frames()]


class TestInterface(RNodeInterface):
    """RNodeInterface with its two hardware-touching paths replaced."""

    def open_port(self):
        self.serial = FakeSerial()

    def configure_device(self):
        # The tests drive initRadio() and validateRadioState() themselves.
        pass


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


def make(**overrides):
    cfg = ConfigObj()
    cfg["name"]            = "TEST"
    cfg["port"]            = "/dev/null"
    cfg["frequency"]       = "869525000"
    cfg["bandwidth"]       = "250000"
    cfg["txpower"]         = "22"
    cfg["spreadingfactor"] = "7"
    cfg["codingrate"]      = "5"
    for k, v in overrides.items():
        cfg[k] = v
    return TestInterface(None, cfg)


LORA_INIT_SEQUENCE = [
    KISS.CMD_FREQUENCY,
    KISS.CMD_BANDWIDTH,
    KISS.CMD_TXPOWER,
    KISS.CMD_SF,
    KISS.CMD_CR,
    KISS.CMD_RADIO_STATE,
]


def test_lora_unchanged():
    print("LoRa (no `modem` key) — backwards compatibility")
    iface = make()
    check("modem defaults to LoRa", iface.modem == KISS.MODEM_LORA, repr(iface.modem))
    check("modem_name is 'lora'", iface.modem_name == "lora", repr(iface.modem_name))
    check("str() is unchanged", str(iface) == "RNodeInterface[TEST]", str(iface))

    iface.initRadio()
    cmds = iface.serial.commands()
    check("initRadio command sequence is unchanged",
          cmds == LORA_INIT_SEQUENCE, cmds)
    check("no CMD_MODEM frame is emitted", KISS.CMD_MODEM not in cmds, cmds)
    check("no CMD_MODEM_PARAM frame is emitted (L1)",
          KISS.CMD_MODEM_PARAM not in cmds, cmds)
    check("no written frame uses the 0x10 command byte (L1)",
          all(w[1] != KISS.CMD_MODEM_PARAM for w in iface.serial.writes if len(w) > 1),
          iface.serial.writes)
    check("no LoRa-only key is rejected by the new validation (L2)",
          iface.modem_bitrate == None and iface.whitening == False,
          (iface.modem_bitrate, iface.whitening))

    iface.r_sf = 7
    iface.r_cr = 5
    iface.r_bandwidth = 250000
    iface.updateBitrate()
    check("bitrate still derived from the LoRa parameters",
          abs(iface.bitrate - 10937.5) < 1.0, iface.bitrate)


def test_gfsk():
    print("(G)FSK (modem = gfsk)")
    iface = make(modem="gfsk", bandwidth="467000")
    check("modem is the GFSK wire value", iface.modem == KISS.MODEM_GFSK, repr(iface.modem))
    check("str() carries the modem", str(iface) == "RNodeInterface[TEST/GFSK]", str(iface))

    iface.initRadio()
    frames = iface.serial.frames()
    cmds = [c for c, _ in frames]
    check("the LoRa parameters are still sent (lock-predicate compatibility)",
          cmds[:6] == LORA_INIT_SEQUENCE, cmds)
    check("CMD_MODEM is sent", KISS.CMD_MODEM in cmds, cmds)
    check("CMD_MODEM is sent right after RADIO_STATE_ON",
          cmds[6] == KISS.CMD_MODEM and cmds[5] == KISS.CMD_RADIO_STATE, cmds)
    check("CMD_MODEM payload is the GFSK wire value",
          frames[6][1] == bytes([KISS.MODEM_GFSK]), frames[6])

    iface.updateBitrate()
    check("bitrate is the resolved GFSK profile rate", iface.bitrate == 150000, iface.bitrate)


def test_fsk():
    print("plain 2-FSK (modem = fsk)")
    iface = make(modem="fsk", bandwidth="467000")
    check("modem is the FSK wire value", iface.modem == KISS.MODEM_FSK, repr(iface.modem))
    iface.initRadio()
    frames = iface.serial.frames()
    check("CMD_MODEM payload is the FSK wire value",
          frames[6][1] == bytes([KISS.MODEM_FSK]), frames[6])
    iface.updateBitrate()
    check("bitrate is the resolved FSK profile rate", iface.bitrate == 300000, iface.bitrate)


def test_config_validation():
    print("configuration validation")

    try:
        make(bandwidth="1")
        check("a LoRa-invalid bandwidth is still rejected in LoRa", False, "no exception")
    except ValueError:
        check("a LoRa-invalid bandwidth is still rejected in LoRa", True)

    try:
        make(modem="gfsk", bandwidth="1")
        check("a bandwidth that is not a (G)FSK table entry is rejected", False, "no exception")
    except ValueError:
        check("a bandwidth that is not a (G)FSK table entry is rejected", True)

    iface = make(modem="gfsk", bandwidth="0", spreadingfactor="0", codingrate="0")
    check("absent LoRa parameters get inert defaults in (G)FSK",
          (iface.bandwidth, iface.sf, iface.cr) == (467000, 7, 5),
          (iface.bandwidth, iface.sf, iface.cr))
    check("absent (G)FSK keys resolve to the pinned GFSK defaults",
          (iface.modem_bitrate, iface.modem_deviation, iface.whitening) == (150000, 75000, False),
          (iface.modem_bitrate, iface.modem_deviation, iface.whitening))

    try:
        make(modem="bogus")
        check("an unknown modem is rejected", False, "no exception")
    except ValueError:
        check("an unknown modem is rejected", True)

    try:
        make(modem="gfsk", bandwidth="467000", frequency="0")
        check("frequency is validated in every modem", False, "no exception")
    except ValueError:
        check("frequency is validated in every modem", True)

    try:
        make(modem="gfsk", bandwidth="467000", txpower="99")
        check("TX power is validated in every modem", False, "no exception")
    except ValueError:
        check("TX power is validated in every modem", True)


def test_validate_radio_state():
    print("validateRadioState")

    iface = make(modem="gfsk", bandwidth="467000")
    iface.serial = FakeSerial()
    iface.r_frequency = 869525000
    iface.r_txpower = 22
    iface.r_state = KISS.RADIO_STATE_ON
    iface.state = KISS.RADIO_STATE_ON
    iface.r_bandwidth = 125000   # deliberately "wrong" for LoRa
    iface.r_sf = 12              # deliberately "wrong" for LoRa
    check("(G)FSK ignores bandwidth and spreading factor mismatches",
          iface.validateRadioState() == True)

    iface.r_txpower = 20
    check("(G)FSK still catches a TX power mismatch",
          iface.validateRadioState() == False)

    iface.r_txpower = 22
    iface.r_state = KISS.RADIO_STATE_OFF
    check("(G)FSK still catches a radio state mismatch",
          iface.validateRadioState() == False)

    iface.r_state = KISS.RADIO_STATE_ON
    iface.r_modem = KISS.MODEM_FSK
    check("(G)FSK catches a modem mismatch",
          iface.validateRadioState() == False)

    lora = make()
    lora.serial = FakeSerial()
    lora.r_frequency = 869525000
    lora.r_txpower = 22
    lora.r_bandwidth = 125000
    lora.r_sf = 7
    lora.r_state = KISS.RADIO_STATE_ON
    lora.state = KISS.RADIO_STATE_ON
    check("LoRa still catches a bandwidth mismatch",
          lora.validateRadioState() == False)


def _stat_writes(serial, command):
    """How many times the exact request frame for `command` was written."""
    return serial.writes.count(bytes([KISS.FEND, command, KISS.FEND]))


def test_stats_polling_off():
    print("statistics polling disabled (default)")
    iface = make()
    check("poll_stats defaults to off", iface.poll_stats == False, repr(iface.poll_stats))

    iface.initRadio()
    cmds = iface.serial.commands()
    check("no CMD_STAT_RX frame is emitted", KISS.CMD_STAT_RX not in cmds, cmds)
    check("no CMD_STAT_TX frame is emitted", KISS.CMD_STAT_TX not in cmds, cmds)
    check("initRadio command sequence is unchanged",
          cmds == LORA_INIT_SEQUENCE, cmds)

    # Even an explicit start must not do anything unless the user opted in.
    iface.start_stats_polling()
    check("start_stats_polling() is a no-op when polling is off",
          iface.should_poll_stats == False, iface.should_poll_stats)
    check("no statistics frame is emitted after start_stats_polling()",
          _stat_writes(iface.serial, KISS.CMD_STAT_RX) == 0
          and _stat_writes(iface.serial, KISS.CMD_STAT_TX) == 0,
          iface.serial.writes)


def test_stats_polling_on():
    print("statistics polling enabled (poll_stats = true)")
    iface = make(poll_stats="true")
    check("poll_stats is parsed from the config", iface.poll_stats == True, repr(iface.poll_stats))

    iface.initRadio()
    check("enabling polling does not change initRadio",
          iface.serial.commands() == LORA_INIT_SEQUENCE, iface.serial.commands())

    # Trigger a request synchronously, so these assertions cannot race the
    # polling thread.
    iface.request_stats()
    frames = iface.serial.frames()
    check("CMD_STAT_RX and CMD_STAT_TX are requested, in that order",
          [c for c, _ in frames[-2:]] == [KISS.CMD_STAT_RX, KISS.CMD_STAT_TX], frames[-2:])
    check("the request frames carry no payload",
          [p for _, p in frames[-2:]] == [b"", b""], frames[-2:])
    check("the raw request bytes are FEND/CMD/FEND",
          iface.serial.writes[-2:] == [bytes([KISS.FEND, KISS.CMD_STAT_RX, KISS.FEND]),
                                       bytes([KISS.FEND, KISS.CMD_STAT_TX, KISS.FEND])],
          iface.serial.writes[-2:])

    # Now the polling thread itself. The interval is shortened and the recorded
    # writes are polled with a bounded wait, so the check is deterministic
    # rather than a sleep-race.
    before = _stat_writes(iface.serial, KISS.CMD_STAT_RX)
    iface.stats_poll_interval = 0.01
    iface.start_stats_polling()
    check("start_stats_polling() starts the poller", iface.should_poll_stats == True,
          iface.should_poll_stats)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if _stat_writes(iface.serial, KISS.CMD_STAT_RX) - before >= 2:
            break
        time.sleep(0.01)
    iface.stop_stats_polling()

    check("the polling thread requests the counters repeatedly",
          _stat_writes(iface.serial, KISS.CMD_STAT_RX) - before >= 2,
          iface.serial.writes)
    check("the polling thread requests both counters",
          _stat_writes(iface.serial, KISS.CMD_STAT_TX) - before >= 2,
          iface.serial.writes)
    check("stop_stats_polling() clears the run flag",
          iface.should_poll_stats == False, iface.should_poll_stats)


def test_locked_radio_frame():
    """The device's dedicated "radio locked" error (0x08) must produce exactly
    one actionable message, and must NOT schedule a reconnect.

    A pty stands in for the serial port, so the real readLoop() parses a real
    KISS frame. This is the only way to exercise the branch without hardware:
    a well-formed interface configuration always sends a frequency before
    commanding the radio on, so it can never leave a device locked.
    """
    print("ERROR_RADIO_LOCKED (0x08) handling in readLoop")
    master, slave = pty.openpty()
    iface = make()
    iface.serial = serial.Serial(os.ttyname(slave), 115200, timeout=0)

    logs = []
    orig_log = RNS.log
    RNS.log = lambda msg, *a, **kw: logs.append(str(msg))
    try:
        t = threading.Thread(target=iface.readLoop, daemon=True)
        t.start()
        time.sleep(0.3)
        os.write(master, bytes([KISS.FEND, KISS.CMD_ERROR, KISS.ERROR_RADIO_LOCKED, KISS.FEND]))
        time.sleep(0.5)

        check("radio_locked is set", iface.radio_locked == True, iface.radio_locked)
        check("the lock is recorded in hw_errors",
              any(e.get("error") == KISS.ERROR_RADIO_LOCKED for e in iface.hw_errors), iface.hw_errors)
        check("fatal_config_error is set", iface.fatal_config_error == True)

        locked = [m for m in logs if "locked" in m.lower()]
        check("exactly one actionable message about the lock", len(locked) == 1, locked)

        # Closing the port must not schedule a reconnect while locked.
        iface.serial.close()
        time.sleep(0.8)
        check("no reconnect is attempted while locked", iface.reconnecting == False, iface.reconnecting)
        check("readLoop exited", not t.is_alive(), t.is_alive())
    finally:
        RNS.log = orig_log
        try: iface.serial.close()
        except Exception: pass
        try: os.close(master)
        except Exception: pass
        try: os.close(slave)
        except Exception: pass


def test_profile_defaults():
    print("(G)FSK/FSK profile keys and pinned defaults")

    check("MODEM_DEFAULT_BANDWIDTH is now the (G)FSK RX bandwidth 467000",
          RNodeInterface.MODEM_DEFAULT_BANDWIDTH == 467000,
          RNodeInterface.MODEM_DEFAULT_BANDWIDTH)
    check("467000 is a table entry",
          467000 in RNodeInterface.GFSK_RX_BANDWIDTHS,
          RNodeInterface.GFSK_RX_BANDWIDTHS)
    check("the table has 21 entries",
          len(RNodeInterface.GFSK_RX_BANDWIDTHS) == 21,
          len(RNodeInterface.GFSK_RX_BANDWIDTHS))

    gfsk = make(modem="gfsk", bandwidth="467000")
    check("absent GFSK keys resolve to the pinned profile",
          (gfsk.bandwidth, gfsk.modem_bitrate, gfsk.modem_deviation, gfsk.whitening)
          == (467000, 150000, 75000, False),
          (gfsk.bandwidth, gfsk.modem_bitrate, gfsk.modem_deviation, gfsk.whitening))

    fsk = make(modem="fsk", bandwidth="467000")
    check("absent FSK keys resolve to the pinned profile",
          (fsk.bandwidth, fsk.modem_bitrate, fsk.modem_deviation, fsk.whitening)
          == (467000, 300000, 75000, False),
          (fsk.bandwidth, fsk.modem_bitrate, fsk.modem_deviation, fsk.whitening))

    parsed = make(modem="gfsk", bandwidth="234300", bitrate="100000",
                  deviation="50000", whitening="true")
    check("the new keys parse",
          (parsed.modem_bitrate, parsed.modem_deviation, parsed.whitening)
          == (100000, 50000, True),
          (parsed.modem_bitrate, parsed.modem_deviation, parsed.whitening))
    check("whitening parses to a real bool", parsed.whitening is True, repr(parsed.whitening))


def _make_capture(**overrides):
    """make(), capturing the log lines, tolerating a ValueError."""
    logs = []
    orig_log = RNS.log
    RNS.log = lambda msg, *a, **kw: logs.append(str(msg))
    try:
        return make(**overrides), logs, None
    except ValueError as e:
        return None, logs, e
    finally:
        RNS.log = orig_log


def test_profile_validation():
    print("(G)FSK/FSK profile validation (H1-H5, A1-A2)")

    iface, logs, err = _make_capture(modem="gfsk", bandwidth="467000", bitrate="400000")
    check("H1 rejects a bitrate above 300000",
          iface is None and any("bitrate" in m.lower() for m in logs), logs)

    iface, logs, err = _make_capture(modem="gfsk", bandwidth="467000", bitrate="500")
    check("H1 rejects a bitrate below 600",
          iface is None and any("bitrate" in m.lower() for m in logs), logs)

    iface, logs, err = _make_capture(modem="gfsk", bandwidth="467000", deviation="250000")
    check("H2 rejects a deviation above 200000",
          iface is None and any("deviation" in m.lower() for m in logs), logs)

    iface, logs, err = _make_capture(modem="gfsk", bandwidth="467000", deviation="500")
    check("H2 rejects a deviation below 600",
          iface is None and any("deviation" in m.lower() for m in logs), logs)

    # The make() helper's default bandwidth is 250000, the LoRa default.
    iface, logs, err = _make_capture(modem="gfsk")
    check("H3 rejects a bandwidth that is not an exact table entry",
          iface is None and any("not one of the supported table entries" in m for m in logs), logs)

    iface, logs, err = _make_capture(modem="gfsk", bandwidth="312500", bitrate="100000", deviation="50000")
    check("H3 rejects a value between two table entries",
          iface is None and any("not one of the supported table entries" in m for m in logs), logs)

    iface, logs, err = _make_capture(modem="gfsk", bandwidth="234300", bitrate="100000", deviation="75000")
    check("H4 rejects 2*Fdev + BR >= BWF",
          iface is None and any("must be less than the RX bandwidth" in m for m in logs), logs)

    iface, logs, err = _make_capture(modem="gfsk", bandwidth="467000", bitrate="300000", deviation="60000")
    check("H5 rejects h = 2*Fdev/BR < 0.5",
          iface is None and any("modulation index" in m for m in logs), logs)

    iface, logs, err = _make_capture(modem="gfsk", bandwidth="234300", bitrate="100000", deviation="50000")
    check("a legal non-pinned profile is accepted", iface is not None, (iface, logs))

    iface, logs, err = _make_capture(modem="fsk", bandwidth="234300", bitrate="100000", deviation="50000")
    check("the same legal profile is accepted in FSK", iface is not None, (iface, logs))

    # A1: GFSK h < 1.0 warns but does not reject.
    iface, logs, err = _make_capture(modem="gfsk", bandwidth="467000", bitrate="200000", deviation="75000")
    check("A1 warns (GFSK h < 1.0) without rejecting",
          iface is not None and any("eye may close" in m for m in logs), (iface, logs))

    # A1 is GFSK-only: the same numbers in plain FSK must not warn.
    iface, logs, err = _make_capture(modem="fsk", bandwidth="467000", bitrate="200000", deviation="75000")
    check("A1 does not fire for plain FSK",
          iface is not None and not any("eye may close" in m for m in logs), logs)

    # A2: the pinned FSK profile sits at 450/467 = 96.4% of the RX bandwidth,
    # but it is the empirically proven default, so A2 is deliberately
    # suppressed for it -- a warning that fires on every default bring-up is
    # one users learn to ignore. A thin *non-pinned* profile still warns.
    iface, logs, err = _make_capture(modem="fsk", bandwidth="467000")
    check("A2 does not warn for the pinned FSK default",
          iface is not None and not any("crystal drift" in m for m in logs), (iface, logs))

    # Non-pinned and thin: BR 100000, Fdev 100000, BW 312000 -> H4 gives
    # 300000 < 312000 (legal) but 300000 > 0.9*312000, so A2 applies.
    iface, logs, err = _make_capture(modem="fsk", bandwidth="312000",
                                     bitrate="100000", deviation="100000")
    check("A2 warns for a thin non-pinned profile without rejecting",
          iface is not None and any("crystal drift" in m for m in logs), (iface, logs))


def test_profile_frames():
    print("(G)FSK profile frames (Appendix A worked example)")

    iface = make(modem="gfsk", bandwidth="234300", bitrate="100000",
                 deviation="50000", whitening="true")
    iface.initRadio()

    expected = [
        bytes([0xC0, 0x10, 0x00, 0x00, 0x01, 0x86, 0xA0, 0xC0]),  # bitrate 100000
        bytes([0xC0, 0x10, 0x01, 0x00, 0x00, 0xC3, 0x50, 0xC0]),  # deviation 50000
        bytes([0xC0, 0x10, 0x02, 0x00, 0x03, 0x93, 0x3C, 0xC0]),  # RX BW 234300
        bytes([0xC0, 0x10, 0x03, 0x01, 0xC0]),                    # whitening on
        bytes([0xC0, 0x10, 0xFF, 0xC0]),                          # query all
    ]
    check("the four set frames and the query-all match Appendix A byte-for-byte",
          iface.serial.writes[-5:] == expected, iface.serial.writes[-5:])

    frames = iface.serial.frames()
    ids = [p[0] for c, p in frames if c == KISS.CMD_MODEM_PARAM]
    check("the sub-parameter IDs are sent in documented order 00 01 02 03 FF",
          ids == [0x00, 0x01, 0x02, 0x03, 0xFF], ids)
    check("the query-all frame is sent after the four set frames",
          [c for c, _ in frames][-1] == KISS.CMD_MODEM_PARAM
          and frames[-1][1] == bytes([KISS.MODEM_PARAM_QUERY_ALL]),
          frames[-1])
    check("CMD_MODEM precedes every CMD_MODEM_PARAM frame",
          [c for c, _ in frames].index(KISS.CMD_MODEM)
          < [c for c, _ in frames].index(KISS.CMD_MODEM_PARAM),
          [c for c, _ in frames])

    pinned = make(modem="gfsk", bandwidth="467000")
    pinned.initRadio()
    check("the pinned GFSK default profile emits 150000/75000/467000/off",
          pinned.serial.writes[-5:] == [
              bytes([0xC0, 0x10, 0x00, 0x00, 0x02, 0x49, 0xF0, 0xC0]),
              bytes([0xC0, 0x10, 0x01, 0x00, 0x01, 0x24, 0xF8, 0xC0]),
              bytes([0xC0, 0x10, 0x02, 0x00, 0x07, 0x20, 0x38, 0xC0]),
              bytes([0xC0, 0x10, 0x03, 0x00, 0xC0]),
              bytes([0xC0, 0x10, 0xFF, 0xC0]),
          ], pinned.serial.writes[-5:])


def test_profile_readback():
    print("(G)FSK profile readback: request vs echo (§1.6)")

    iface = make(modem="gfsk", bandwidth="467000")
    iface.serial = FakeSerial()

    iface.queryModemProfile()
    check("the query-all frame is C0 10 FF C0",
          iface.serial.writes == [bytes([0xC0, 0x10, 0xFF, 0xC0])],
          iface.serial.writes)

    # A device that echoes exactly what was requested: accepted.
    iface.queryModemProfile = lambda: setattr(iface, "r_modem_param", (150000, 75000, 467000, 0))
    iface.verifyModemProfile()
    check("a matching readback is accepted", iface.modem_param_mismatch == False,
          iface.modem_param_mismatch)

    # A device that echoes something different (whitening flipped): failure.
    iface.queryModemProfile = lambda: setattr(iface, "r_modem_param", (150000, 75000, 467000, 1))
    iface.verifyModemProfile()
    check("a differing readback is treated as failure", iface.modem_param_mismatch == True,
          iface.modem_param_mismatch)

    # A device that does not answer: the compatibility path, not a failure.
    iface.MODEM_PARAM_TIMEOUT = 0.05
    iface.queryModemProfile = lambda: None
    iface.verifyModemProfile()
    check("no answer takes the compatibility path without failing",
          iface.modem_param_mismatch == False, iface.modem_param_mismatch)


def test_modem_param_parser():
    print("CMD_MODEM_PARAM whole-profile echo parsing in readLoop")
    master, slave = pty.openpty()
    iface = make(modem="gfsk", bandwidth="467000")
    iface.serial = serial.Serial(os.ttyname(slave), 115200, timeout=0)

    orig_log = RNS.log
    RNS.log = lambda msg, *a, **kw: None
    try:
        t = threading.Thread(target=iface.readLoop, daemon=True)
        t.start()
        time.sleep(0.3)
        # Appendix A query-all reply: br 100000, fdev 50000, bw 234300, wh on.
        reply = bytes([KISS.FEND, KISS.CMD_MODEM_PARAM, 0xFF,
                       0x00, 0x01, 0x86, 0xA0,
                       0x00, 0x00, 0xC3, 0x50,
                       0x00, 0x03, 0x93, 0x3C,
                       0x01, KISS.FEND])
        os.write(master, reply)
        time.sleep(0.4)
        check("the whole-profile echo is parsed",
              iface.r_modem_param == (100000, 50000, 234300, 1), iface.r_modem_param)

        # The same reply, but with values that contain FEND/FESC bytes, so the
        # device must escape them (§1.4): br 0x0000C000, fdev 0x0000DB00.
        iface.r_modem_param = None
        escaped = bytes([KISS.FEND, KISS.CMD_MODEM_PARAM, 0xFF,
                         0x00, 0x00, 0xDB, 0xDC, 0x00,
                         0x00, 0x00, 0xDB, 0xDD, 0x00,
                         0x00, 0x07, 0x20, 0x38,
                         0x00, KISS.FEND])
        os.write(master, escaped)
        time.sleep(0.4)
        check("an escaped whole-profile echo is parsed",
              iface.r_modem_param == (49152, 56064, 467000, 0), iface.r_modem_param)

        iface.detached = True
        iface.serial.close()
        time.sleep(0.4)
        check("readLoop exited", not t.is_alive(), t.is_alive())
    finally:
        RNS.log = orig_log
        try: iface.serial.close()
        except Exception: pass
        try: os.close(master)
        except Exception: pass
        try: os.close(slave)
        except Exception: pass


def test_bad_modem_param_capability():
    print("ERROR_BAD_MODEM (0x07) as the CMD_MODEM_PARAM capability probe")
    master, slave = pty.openpty()
    iface = make(modem="gfsk", bandwidth="467000")
    iface.serial = serial.Serial(os.ttyname(slave), 115200, timeout=0)

    logs = []
    orig_log = RNS.log
    RNS.log = lambda msg, *a, **kw: logs.append(str(msg))
    try:
        t = threading.Thread(target=iface.readLoop, daemon=True)
        t.start()
        time.sleep(0.3)

        # The modem switch was acknowledged, so a 0x07 answers a profile frame:
        # the device predates CMD_MODEM_PARAM. Log and continue.
        iface.r_modem = iface.modem
        iface.expecting_modem_param = True
        os.write(master, bytes([KISS.FEND, KISS.CMD_ERROR, KISS.ERROR_BAD_MODEM, KISS.FEND]))
        time.sleep(0.4)
        check("0x07 to a profile frame does not set modem_error",
              iface.modem_error == False, iface.modem_error)
        check("0x07 to a profile frame is recorded as unsupported",
              iface.modem_param_unsupported == True, iface.modem_param_unsupported)
        check("the capability message is logged clearly",
              any("does not implement CMD_MODEM_PARAM" in m for m in logs), logs)
        check("the interface is not treated as a fatal config error",
              iface.fatal_config_error == False, iface.fatal_config_error)

        # No modem switch was acknowledged: a 0x07 is the modem error, as before.
        iface.expecting_modem_param = False
        iface.r_modem = None
        os.write(master, bytes([KISS.FEND, KISS.CMD_ERROR, KISS.ERROR_BAD_MODEM, KISS.FEND]))
        time.sleep(0.4)
        check("0x07 without an acknowledged modem switch still sets modem_error",
              iface.modem_error == True, iface.modem_error)

        iface.serial.close()
        time.sleep(0.4)
        check("readLoop exited", not t.is_alive(), t.is_alive())
    finally:
        RNS.log = orig_log
        try: iface.serial.close()
        except Exception: pass
        try: os.close(master)
        except Exception: pass
        try: os.close(slave)
        except Exception: pass


def main():
    for t in (test_lora_unchanged, test_gfsk, test_fsk,
              test_config_validation, test_validate_radio_state,
              test_profile_defaults, test_profile_validation,
              test_profile_frames, test_profile_readback,
              test_modem_param_parser, test_bad_modem_param_capability,
              test_stats_polling_off, test_stats_polling_on,
              test_locked_radio_frame):
        t()

    print()
    if FAILURES:
        print("%d check(s) FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
