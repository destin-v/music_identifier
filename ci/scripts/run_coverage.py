from os import system

# Run coverage using pytest, then record results to docs.
system("coverage run -m pytest")
system("coverage html")
