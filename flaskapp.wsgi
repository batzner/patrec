#!/usr/bin/python
                        
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/html/patrec/")

from src import app as application
if __name__ == "__main__":
    application.run()
application.secret_key = '9(9zp!52ngtw2b0w2vfn-twoz4c+2gaa71234fql'
