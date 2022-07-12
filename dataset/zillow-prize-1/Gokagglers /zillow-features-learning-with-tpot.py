# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "_cell_guid": "0cfcf2a8-a243-16b4-03fe-5ad68963de0a"
      },
      "source": [
        "TPOT is an automated model selection library.  Here is some of the code it writes."
      ]
    },
    {
      "cell_type": "code",
      
      "metadata": {
        "_cell_guid": "da288898-0347-b07b-5bba-dd67a2286645"
      },
      "outputs": [],
      "source": [
        "# This Python 3 environment comes with many helpful analytics libraries installed\n",
        "# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python\n",
        "# For example, here's several helpful packages to load in \n",
        "\n",
        "import numpy as np # linear algebra\n",
        "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
        "\n",
        "# Input data files are available in the \"../input/\" directory.\n",
        "# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory\n",
        "\n",
        "from subprocess import check_output\n",
        "print(check_output([\"ls\", \"../input\"]).decode(\"utf8\"))\n",
        "\n",
        "# Any results you write to the current directory are saved as output."
      ]
    },
    {
      "cell_type": "code",
      
      "metadata": {
        "_cell_guid": "4c0954f5-3352-b643-d850-3e0f018a096c"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "\n",
        "from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.pipeline import make_pipeline, make_union\n",
        "from sklearn.preprocessing import FunctionTransformer\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "# NOTE: Make sure that the class is labeled 'class' in the data file\n",
        "df = pd.read_csv('../input/glass.csv')\n",
        "\n",
        "features = df.drop('Type', axis = 1)\n",
        "training_features, testing_features, training_classes, testing_classes = \\\n",
        "    train_test_split(features, df['Type'], random_state=42)\n",
        "\n",
        "exported_pipeline = make_pipeline(\n",
        "    make_union(\n",
        "        FunctionTransformer(lambda X: X),\n",
        "        FunctionTransformer(lambda X: X)\n",
        "    ),\n",
        "    make_union(VotingClassifier([(\"est\", GaussianNB())]), FunctionTransformer(lambda X: X)),\n",
        "    ExtraTreesClassifier(criterion=\"entropy\", max_features=0.16, n_estimators=500)\n",
        ")\n",
        "\n",
        "exported_pipeline.fit(training_features, training_classes)\n",
        "results = exported_pipeline.predict(testing_features)\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "_cell_guid": "e9a89d5e-95f7-f09f-8a6d-d745379f98eb"
      },
      "outputs": [],
      "source": [
        "print(accuracy_score(testing_classes, results))"
      ]
    },
    {
      "cell_type": "code",
      
      "metadata": {
        "_cell_guid": "53534300-cc63-9ffc-304b-a4d7040e44ac"
      },
      "outputs": [],
      "source": ""
    }
  ],
  "metadata": {
    "_change_revision": 0,
    
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.5.2"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}