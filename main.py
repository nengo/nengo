import nengo_gui.swi
from nengo_gui.server import NengoGui

import optparse
parser = optparse.OptionParser()
parser.add_option('-p', '--password', dest='password', metavar='PASS',
                  default=None, help='password for remote access')
parser.add_option('-r', '--refresh', dest='refresh', metavar='TIME',
                  default=0, help='interval to check server for changes',
                  type='int')
parser.add_option('-P', '--port', dest='port', metavar='PORT',
                  default=8080, type='int', help='port to run server on')
(options, args) = parser.parse_args()

NengoGui.set_refresh_interval(options.refresh)

if len(args) > 0:
    NengoGui.set_default_filename(args[0])

addr = 'localhost'
if options.password is not None:
    nengo_gui.swi.addUser('', options.password)
    addr = ''   # allow connections from anywhere
else:
    nengo_gui.swi.browser(options.port)

nengo_gui.swi.start(NengoGui, options.port, addr=addr)
