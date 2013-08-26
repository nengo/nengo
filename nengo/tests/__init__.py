
# -- putting this in a separate file helps prevent circular import madness
from list_of_test_modules import simulator_test_case_mods

for test_module in simulator_test_case_mods:
    __import__(test_module)

