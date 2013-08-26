
# -- import these modules to populate the registry
#    of standard simulator test cases
#    `helpers.simulator_test_cases`
simulator_test_case_mods = [
    'nengo.tests.test_circularconv',
    'nengo.tests.test_ensemble',
    ]
for test_module in simulator_test_case_mods:
    __import__(test_module)

