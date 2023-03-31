import os

# Setup the terminal size because the output file is tied to terminal sizing.
os.environ["LINES"] = "25"
os.environ["COLUMNS"] = "200"

# Profile the code.  For additional options see $ scalene -h
os.system("scalene --outfile ./docs/profile.html --html main.py")
