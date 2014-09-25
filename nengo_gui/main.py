import nengo_gui.swi
from nengo_gui.server import NengoGui

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--password', dest='password', metavar='PASS',
        help='password for remote access')
    parser.add_argument(
        '-r', '--refresh', dest='refresh', metavar='TIME',
        default=0, help='interval to check server for changes', type=int)
    parser.add_argument(
        '-s', '--simulator', dest='simulator', metavar='SIM',
        default="nengo", type=str, help='simulator platform')
    parser.add_argument(
        '-P', '--port', dest='port', metavar='PORT',
        default=8080, type=int, help='port to run server on')
    parser.add_argument(
        'filename', nargs='?', type=str, help='initial file to load')
    args = parser.parse_args()

    NengoGui.set_refresh_interval(args.refresh)
    NengoGui.set_simulator_class(__import__(args.simulator).Simulator)
    if args.simulator in ['nengo_spinnaker']:
        # TODO: Simulators should have a supports_realtime() flag
        NengoGui.set_realtime_simulator_mode(True)

    if args.filename is not None:
        NengoGui.set_default_filename(args.filename)

    addr = 'localhost'
    if args.password is not None:
        nengo_gui.swi.addUser('', args.password)
        addr = ''   # allow connections from anywhere
    else:
        nengo_gui.swi.browser(args.port)

    nengo_gui.swi.start(NengoGui, args.port, addr=addr)

if __name__ == '__main__':
    main()
