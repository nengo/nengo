# Download Anoconda's various eersions into various folders
# TODO: Make it multi-threaded

import os
import urllib, urlparse

# Because I want pretty dict initialization
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

# TODO: Figure out how to just get the most recent version and then dynamically
# create the batch file and shell file
# Where the get the various packages
anaconda_list = Vividict()
anaconda_list["os_x"]["64_bit"] = "http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-MacOSX-x86_64.pkg"
anaconda_list["linux"]["32_bit"] = "http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Linux-x86.sh"
anaconda_list["linux"]["64_bit"] = "http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Linux-x86_64.sh"
anaconda_list["windows"]["32_bit"] = "http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Windows-x86.exe"
anaconda_list["windows"]["64_bit"] = "http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Windows-x86_64.exe"

packages_list = Vividict()
#packages_list["linux"]
#packages_list = ["numpy", "ipython notebook", "Tornado"]

# Make directories and download packages
for op_sys, op_key in anaconda_list.iteritems():
    for arch, url in op_key.iteritems():
        print("%s %s %s" %(op_sys, arch, url))
        target_dir = "packages/%s/%s/" %(op_sys, arch,)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print("downloading %s" %url)
        print("target %s/%s" %(target_dir, urlparse.urlsplit(url).path.split("/")[-1],))
        urllib.urlretrieve(url, "%s/%s" %(target_dir, urlparse.urlsplit(url).path.split("/")[-1],))

# Build the docs # TODO: Add offline README indicating where the offline docs are