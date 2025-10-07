import {PyrightWrapper} from "../src/check_hallucination.ts";


describe("test hallucination detector", () => {
    const path = "/opt/homebrew/anaconda3/envs/bigcodebench/bin/python";
    //const path = "/home/ec2-user/anaconda3/envs/bigcodebench/bin/python";
    const wrapper = new PyrightWrapper(path, true);

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
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("index with numbers", () => {
        expect(wrapper.detectHallucinations({
            left_context: "import numpy as nk\n",
            generation: "a = nk200",
            right_context: "",
            is_end: false
        })).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
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
            hallucination: false,
            index_of_hallucination: null
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
            hallucination: false,
            index_of_hallucination: null
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
            hallucination: false,
            index_of_hallucination: null
        })
    })


    test("don't allow sketchy assign", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "x v = nk.dot(",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
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

    test("hallucinated numpy function with wrong prefix", () => {
        const input =
            {
                "left_context": "import numpy as nk\n\ndef main():\n    # Create a random 5x5 numpy array\n    ",
                "right_context": "",
                "generation": "partitioner = np.MaxDivisionPartitioner(4)",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 15
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


    test("allow incomplete generator", () => {
        const input =
            {
                "left_context": "",
                "right_context": "",
                "generation": "[foo",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("ban complete generator", () => {
        const input =
            {
                "left_context": "",
                "right_context": "",
                "generation": "[foo.x for bar in []]",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 3
        })
    })

    test("allow unbound if in prefix", () => {
        const input =
            {
                "left_context": "foo.",
                "right_context": "",
                "generation": "bar",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("pyplot xlabel", () => {
        const input =
            {
                "left_context": "import pandas as pd\n" +
                    "import matplotlib.pyplot as plt\n" +
                    "import numpy as np\n" +
                    "\n" +
                    "\n" +
                    "def task_func(num_groups=5, data_size=5, labels=None):\n" +
                    "    \"\"\"\n" +
                    "    Generate random data and visualize it with a stacked bar chart, saving the chart to a file.\n" +
                    "    This function facilitates the exploration and sharing of data distribution across multiple categories.\n" +
                    "\n" +
                    "    Parameters:\n" +
                    "    num_groups (int): Number of groups for which data is to be generated, defaulting to 5.\n" +
                    "    data_size (int): Number of data points for each group, defaulting to 5.\n" +
                    "    labels (list of str, optional): Labels for the groups. If None, default labels 'Group1', 'Group2', ...,\n" +
                    "    'GroupN' are generated.\n" +
                    "\n" +
                    "    Returns:\n" +
                    "    tuple: A tuple containing:\n" +
                    "        - matplotlib.figure.Figure: The Figure object containing the stacked bar chart.\n" +
                    "        - pandas.DataFrame: The DataFrame with randomly generated data.\n" +
                    "        - str: The filename where the plot is saved ('test_plot.png').\n" +
                    "\n" +
                    "    Requirements:\n" +
                    "    - pandas\n" +
                    "    - matplotlib\n" +
                    "    - numpy\n" +
                    "\n" +
                    "    Example:\n" +
                    "    >>> np.random.seed(0)\n" +
                    "    >>> fig, data, plot_filename = task_func(3, 3, ['A', 'B', 'C'])\n" +
                    "    >>> print(data)\n" +
                    "              A         B         C\n" +
                    "    0  0.548814  0.715189  0.602763\n" +
                    "    1  0.544883  0.423655  0.645894\n" +
                    "    2  0.437587  0.891773  0.963663\n" +
                    "    >>> print(plot_filename)\n" +
                    "    test_plot.png\n" +
                    "    \"\"\"\n" +
                    "#     num_groups = 5\n" +
                    "#     data_size = 5\n" +
                    "#     labels = None\n" +
                    "    if labels is None:\n" +
                    "        labels = [f\"Group{i}\" for i in range(1, num_groups + 1)]\n" +
                    "\n" +
                    "    # Generate random data\n" +
                    "    data = pd.Series(np.random.random(data_size * num_groups), index=labels * data_size)\n" +
                    "\n" +
                    "    # Visualize the data with stacked bar chart\n" +
                    "    plt_fig, plt_ax = plt.figure(), plt.bar(np.arange(len(labels)), data.values, width=0.3)\n" +
                    "    plt_ax = plt_ax[0]\n" +
                    "    plt.xticks(np.arange(len(labels)), labels)\n" +
                    "    plt.ylabel(\"Value\")\n" +
                    "    plt."
                ,
                "right_context": "",
                "generation": "xlabel",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("BigCodeBench/82", () => {
        const input =
            {
                "left_context": "import os\n" +
                    "import shutil\n" +
                    "from flask_login import login_user\n" +
                    "import sys\n" +
                    "from os import environ\n" +
                    "from flask import Flask, render_template, redirect, url_for\n" +
                    "from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user\n" +
                    "from flask_wtf import FlaskForm\n" +
                    "from wtforms import StringField, PasswordField, SubmitField\n" +
                    "from wtforms.validators import DataRequired, Length\n" +
                    "from werkzeug.security import generate_password_hash, check_password_hash\n" +
                    "\n" +
                    "class LoginForm(FlaskForm):\n" +
                    "    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])\n" +
                    "    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=80)])\n" +
                    "    submit = SubmitField('Log In')\n" +
                    "\n" +
                    "login_manager = LoginManager()\n" +
                    "\n" +
                    "def task_func(secret_key, template_folder):\n" +
                    "    \"\"\"\n" +
                    "    Creates a Flask application with configured user authentication using Flask-Login.\n" +
                    "    It defines routes for login, logout, and a protected page. The user authentication\n" +
                    "    is managed with a simple User class and a login form using Flask-WTF. The application\n" +
                    "    uses dynamic configuration for security and template rendering.\n" +
                    "\n" +
                    "    Parameters:\n" +
                    "        secret_key (str): A secret key for the application to use for session management.\n" +
                    "        template_folder (str): The path to the directory containing Flask templates.\n" +
                    "\n" +
                    "    Requirements:\n" +
                    "    - flask\n" +
                    "    - flask_login\n" +
                    "    - flask_wtf\n" +
                    "    - wtforms\n" +
                    "    - wtforms.validators\n" +
                    "    - werkzeug.security\n" +
                    "\n" +
                    "    Returns:\n" +
                    "        Flask: A Flask application instance configured for user authentication.\n" +
                    "\n" +
                    "    Examples:\n" +
                    "    >>> app = task_func('mysecretkey', 'templates')\n" +
                    "    >>> 'login' in [rule.endpoint for rule in app.url_map.iter_rules()]\n" +
                    "    True\n" +
                    "    >>> app.config['SECRET_KEY'] == 'mysecretkey'\n" +
                    "    True\n" +
                    "    \"\"\"\n" +
                    "\n" +
                    "    class User(UserMixin):\n" +
                    "        def __init__(self, id):\n" +
                    "            self.id = id\n" +
                    "\n" +
                    "    class LoginForm(FlaskForm):\n" +
                    "        username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])\n" +
                    "        password = PasswordField('<PASSWORD>', validators=[DataRequired(), Length(min=8, max=80)])\n" +
                    "        submit = SubmitField('Log In')\n" +
                    "\n" +
                    "    app = Flask(__name__, template_folder=template_folder)\n" +
                    "    app.config['SECRET_KEY'] = secret_key\n" +
                    "\n" +
                    "    @login_manager.user_loader\n" +
                    "    def load_user(user_id):\n" +
                    "        return User(user_id)\n" +
                    "\n" +
                    "    @app.route('/')\n" +
                    "    def index():\n" +
                    "        return redirect(url_for('login'))\n" +
                    "\n" +
                    "    @app.route('/login', methods=['GET', 'POST'])\n" +
                    "    def login():\n" +
                    "        form = LoginForm()",
                "right_context": "",
                "generation": "\n" +
                    "        if form.validate_on_submit():",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: false,
            index_of_hallucination: null
        })
    })

    test("boto3 describe_scaling_plan", () => {
        const input =
            {
                "left_context": "import boto3\n" +
                    "\n" +
                    "# initialize autoscaling-plans client\n" +
                    "client = boto3.client(\"autoscaling-plans\")\n" +
                    "\n" +
                    "# Retrieve capacity forecast for a scalable resource\n" +
                    "response = client."
                ,
                "right_context": "",
                "generation": "describe_scaling_plan_resource_forecast_data(\n" +
                    "    ScalingPlanName=\"string\",\n" +
                    "    ServiceNamespace=\"string\",\n" +
                    "    ResourceId=\"string\",\n" +
                    "    ScalableDimension=\"string\",\n" +
                    "    ForecastDataType=\"LoadForecast\"|\"CapacityForecast\"|\"ScheduledActionMinCapacity\"|\"ScheduledActionMaxCapacity\",\n" +
                    ")",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 30
        })
    })

    test("boto3 put_alternate_contact", () => {
        const input =
            {
                "left_context": "import boto3\n" +
                    "\n" +
                    "# initialize iot client\n" +
                    "client = boto3.client(\"account\")\n" +
                    "\n" +
                    "# Generate a new metric for an IoT device fleet\n" +
                    "response = client."
                ,
                "right_context": "",
                "generation": "modify_alternate_contact(\n" +
                    "    AlternateContactType=\"Billing\",\n" +
                    "    EmailAddress= \"<EMAIL>\",\n" +
                    "    Name=\"<NAME>\",\n" +
                    "    PhoneNumber=\"+12065550100\",\n" +
                    ")",
                "is_end": false
            }
        expect(wrapper.detectHallucinations(input)).toStrictEqual({
            hallucination: true,
            index_of_hallucination: 1
        })
    })

})