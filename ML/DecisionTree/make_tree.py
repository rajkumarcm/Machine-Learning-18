"""-----------------------------------
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Decision Tree using ID3
-----------------------------------"""

import numpy as np
import operator
import pandas as pd

def load_data(file_name, ignore_cols):
    data = pd.read_csv(filepath_or_buffer=file_name,
                       header=0,
                       engine='python',
                       #skipfooter=0,
                       skip_blank_lines=False,
                       error_bad_lines=True)

    cols = data.columns

    filtered_cols = []

    for col in cols:
        if col not in ignore_cols:
            filtered_cols.append(col)

    data = data[filtered_cols]
    filtered_cols = list(filter(lambda x: x != 'Survived', filtered_cols))
    return data, filtered_cols

def quantise_data(data):

    """

    :param data:
    :return:
    """
    # Quantise age variable-------------------------------------------------
    # max_age = int(data['Age'].max())
    max_age = 80  # From training data set
    # Age From 0 till 5 ( Constraint for 5 will be on the next condition)
    indices1 = data['Age'] > 0

    # Age 5 to 10
    indices2 = data['Age'] > 5

    indices = [indices1, indices2]
    for age in range(10, max_age+1, 10):
        tmp = data['Age'] > age
        indices.append(tmp)

    step = int(((max_age - 10) / 10) + 2)
    for i, j in zip(range(0, step -1), range(len(indices))):
        tmp = indices[j]
        data['Age'][tmp] = i
    #-----------------------------------------------------------------------

    # Quantise Fare variable------------------------------------------------
    # convert_dict = {'Fare': int}
    # data = data.astype(convert_dict)
    # max_fare = int(data['Fare'].max())
    max_fare = 512  # Found from training data set

    indices = []
    for fare in range(0, max_fare+1, 20):
        tmp = data['Fare'] > fare
        indices.append(tmp)

    for i in range(len(indices)):
        tmp = indices[i]
        data['Fare'][tmp] = i

    return data

def get_best_col(data, attributes, target_attribute):
    info_gain = {}
    t_length = data[target_attribute].size
    pos_count = data[target_attribute].sum()
    neg_count = t_length - pos_count
    entropy_d = 0
    if pos_count and neg_count:
        entropy_d = pos_count/t_length * np.log2(pos_count/t_length) - \
                    neg_count/t_length * np.log2(neg_count/t_length)
    for col in attributes:
        values = data[col].unique()
        tmp_entropy = entropy_d
        for value in values:
            indices = data[col] == value
            tmp_data = data[target_attribute][indices]

            # Do the following step only when you are dealing with binary classification problem
            # perhaps Decision Tree is a binary classifier
            length = tmp_data.size
            pos_count = tmp_data.sum()
            neg_count = length - pos_count
            if pos_count and neg_count:

                # Here positive count and the negative count represent the number of positive and negative
                # examples of each value of corresponding attribute `col` in attributes
                tmp_entropy -= ((pos_count + neg_count)/t_length) * ((pos_count/length) * np.log2(pos_count/length) +
                                                                    (neg_count/length) * np.log2(neg_count/length))
        info_gain[col] = tmp_entropy
    return max(info_gain.items(), key=operator.itemgetter(1))[0]


def make_tree(data, attributes, target_attribute, label=None):

    if (data[target_attribute].max() - data[target_attribute].min()) == 0:
        val = True
        if not data[target_attribute].iloc[0]:
            val = False
        return {'label': label, 'childNodes': None, 'value': val}

    elif len(attributes) == 0 or len(attributes) == 1:

        # What happens when we are left with no attributes
        # Then what we are to create would be just a leaf
        ans = data[target_attribute].mode()[0]
        val = True
        if not ans:
            val = False
        return {'label': label, 'childNodes': None, 'value': val}

    else:

        # This happens when we are at an intermediate node (branch)
        # which has its own descendant nodes or leaf/leaves
        best_col = get_best_col(data, attributes, target_attribute)
        node = {'label': best_col, 'childNodes': None, 'value': None}
        child_nodes = {}
        un_values = data[best_col].unique()

        # For each value in unique values
        # i.e., under each branch
        for value in un_values:
            # Create a subset of data whose best_column attribute
            # has the value
            sub_data = None
            if np.str(value) != 'nan':
                indices = data[best_col] == value
                sub_data = data[indices]
            else:
                sub_data = data[data[best_col].isnull()]
            if value is np.nan or np.str(value) == 'nan':
                value = np.str('nan')
            sub_attributes = list(filter(lambda x: x != best_col, attributes))
            child_nodes[value] = make_tree(sub_data, sub_attributes, target_attribute, label=value)
        node['childNodes'] = child_nodes
        node['value'] = None
        return node

def infer(data, tree):

    """--------------------------------------------------------------------
    Function that navigate through the decision tree to infer the answer
    :param data: Data whose predictor variable need to be inferred
    :return: inferred value of predictor variable
    ---------------------------------------------------------------------"""

    length = data.shape[0]
    values = []
    for i in range(length):
        sample = data.iloc[i]
        node = tree
        while True:
            if node['childNodes'] is None:
                break
            else:
                nodes = node['childNodes']
                col = node['label']
                branch = sample[col]
                if np.str(branch) == 'nan':
                    branch = np.str('nan')
                # If the branch is present in nodes
                branches = list(nodes.keys())
                if branch in branches:
                    node = nodes[branch]
                else:
                    # Otherwise we just randomly pick a branch
                    n_branches = len(branches)
                    rand_indx = np.random.randint(0, n_branches, 1)
                    branch = branches[rand_indx[0]]
                    node = nodes[branch]

        # The program at this point has reached the leaf node

        values.append(node['value'])
    return values

if __name__ == '__main__':

    """-------------------------------------------
    Load data and build tree on training data set
    --------------------------------------------"""
    ignore_cols = ['PassengerId', 'Name', 'Ticket']
    data, attributes = load_data('/home/rajkumarcm/Documents/tmp/data/train.csv', ignore_cols)
    data = quantise_data(data)
    tree = make_tree(data, attributes, target_attribute='Survived')
    # values = infer(data, tree)
    # Save tree
    # np.save('data/tree.npy', tree)

    """-------------------------------------------
    Load test data, and make inference on it
    -------------------------------------------"""
    # tree = np.load("data/tree.npy")
    data, attributes = load_data('/home/rajkumarcm/Documents/Machine-Learning-18/ML/DecisionTree/data/titanic/test.csv', ignore_cols)
    data = quantise_data(data)
    values = infer(data, tree)
    result = []
    ignore_cols = ['Name', 'Ticket']
    data, attributes = load_data('/home/rajkumarcm/Documents/Machine-Learning-18/ML/DecisionTree/data/titanic/test.csv', ignore_cols)
    for i in range(len(values)):
        id = data['PassengerId'].iloc[i]
        value = values[i]
        tmp = 1
        if not value:
            tmp = 0
        result.append([id, tmp])
    result_df = pd.DataFrame(result, columns=['PassengerId', 'Survived'])
    result_df.to_csv(path_or_buf='/home/rajkumarcm/Documents/Machine-Learning-18/ML/DecisionTree/data/titanic/raj_test_submission.csv',
                     index=False,
                     header=True)
    print("Finished...")
