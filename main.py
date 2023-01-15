import ssl
from menu import Menu

ssl._create_default_https_context = ssl._create_unverified_context

Menu.run()
