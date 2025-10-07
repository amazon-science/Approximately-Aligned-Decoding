import {PyrightWrapper} from "../src/check_hallucination.ts";


describe("test hallucination detector", () => {
    const wrapper = new PyrightWrapper("/opt/homebrew/anaconda3/envs/bigcodebench/bin/python", false);

    test("library function not hallucination", () => {
        expect(wrapper.detectHallucinations({
            left_context: "import re\nx = ",
            generation: "re.compile",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("library function hallucination", () => {
        expect(wrapper.detectHallucinations({
            left_context: "import re\nx = ",
            generation: "re.compare",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 7
        })
    })

    test("imported function good", () => {
        expect(wrapper.detectHallucinations({
            left_context: "import numpy as np\nx = ",
            generation: "np.ma",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("imported function bad", () => {
        expect(wrapper.detectHallucinations({
            left_context: "import numpy as np\nx = ",
            generation: "np.mx",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 4
        })
    })

    test("multiple errors", () => {
        expect(wrapper.detectHallucinations({
            left_context: "",
            generation: "a = x\ny = x",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 4
        })
    })

    test("index with numbers", () => {
        expect(wrapper.detectHallucinations({
            left_context: "import numpy as nk\n",
            generation: "a = nk200",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 6
        })
    })

    test("index with numbers not hallucination", () => {
        expect(wrapper.detectHallucinations({
            left_context: "import numpy as nk\n",
            generation: "a = nk",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("partial for", () => {
        expect(wrapper.detectHallucinations({
            left_context: "",
            generation: "for test",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("replicate issue- not including completions properly", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    random_grid = 5\n\n    n",
                "right_context": "",
                "generation": "k_array = n",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("replicate issue 2", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    a",
                "right_context": "",
                "generation": " = n2d.",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 4
        })
    })

    test("replicate issue 3", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "rand_5x5",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("hallucination in binding position at end", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "rand_5x5.",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 8
        })
    })

    test("not hallucination for complex binding", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "(foo12345, [x92375, *y10384])",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("not hallucination for complex binding incomplete", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "(foo12345, [x92375, *y10384]",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("yes hallucination after dot", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "(foo12345, [x92375, *y10384]).",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 29
        })
    })


    test("don't allow sketchy assign", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "x v = numpy.dot(",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 2
        })
    })

    test("hallucinated numpy function", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "partitioner = nk.MaxDivisionPartitioner(4)",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 17
        })
    })

    test("allow kwargs", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    data = nk.ndarray(nk.generator.uniform.random(5,5), d",
                "right_context": "",
                "generation": "type=nk.",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })


    test("allow kwargs 2", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    data = nk.ndarray(nk.generator.uniform.random(5,5), ",
                "right_context": "",
                "generation": "dtyp",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("allow if", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "if ",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })
    test("allow argparse", () => {
        const input =
            {
                "left_context": "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--foo', type=str)\nargs = parser.parse_args()\n",
                "right_context": "",
                "generation": "print(args.foo)",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })
})