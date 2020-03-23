var clover = new Object();

// JSON: {classes : [{name, id, sl, el,  methods : [{sl, el}, ...]}, ...]}
clover.pageData = {"classes":[{"el":36,"id":39,"methods":[{"el":25,"sc":2,"sl":23},{"el":30,"sc":2,"sl":27},{"el":35,"sc":2,"sl":32}],"name":"RandomVariableFloatFactory","sl":16}]}

// JSON: {test_ID : {"methods": [ID1, ID2, ID3...], "name" : "testXXX() void"}, ...};
clover.testTargets = {"test_12":{"methods":[{"sl":23},{"sl":27},{"sl":32}],"name":"\"{0}\"","pass":true,"statements":[{"sl":24},{"sl":29},{"sl":34}]},"test_22":{"methods":[{"sl":23},{"sl":27},{"sl":32}],"name":"\"{0}\"","pass":true,"statements":[{"sl":24},{"sl":29},{"sl":34}]},"test_9":{"methods":[{"sl":23},{"sl":27},{"sl":32}],"name":"testSwaptionSmileCalibration","pass":true,"statements":[{"sl":24},{"sl":29},{"sl":34}]}}

// JSON: { lines : [{tests : [testid1, testid2, testid3, ...]}, ...]};
clover.srcFileLines = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [22, 9, 12], [22, 9, 12], [], [], [22, 9, 12], [], [22, 9, 12], [], [], [22, 9, 12], [], [22, 9, 12], [], []]