

import numpy as np
import unittest
import pandas as pd
from sqlalchemy import create_engine
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column






# Create a new SQLite database (or connect to an existing one)
engine = create_engine('sqlite:///ideal_functions.db')

# Load the training data
# training_files = ['train1.csv', 'train2.csv', 'train3.csv', 'train4.csv']
# training_dfs = [pd.read_csv(file) for file in training_files]

# Combine training data into a single DataFrame
# combined_training_df = pd.concat([df.set_index('x') for df in training_dfs], axis=1).reset_index()
# combined_training_df.columns = ['x', 'y1', 'y2', 'y3', 'y4']

# Load the ideal functions data
ideal_functions_df = pd.read_csv('ideal.csv')
train_df = pd.read_csv('train.csv')

# Save the DataFrames to the SQLite database
train_df.to_sql('training_data', engine, index=False, if_exists='replace')
ideal_functions_df.to_sql('ideal_functions', engine, index=False, if_exists='replace')

print("Training data and ideal functions loaded into the SQLite database")




def find_ideal_functions(engine):
    training_data = pd.read_sql('training_data', engine)
    ideal_functions = pd.read_sql('ideal_functions', engine)

    x_values = training_data['x'].values
    y_trainings = training_data.iloc[:, 1:].values

    best_functions = []

    for y_training in y_trainings.T:
        min_deviation = float('inf')
        best_function = None

        for column in ideal_functions.columns[1:]:
            y_ideal = ideal_functions[column].values
            deviation = np.sum((y_training - y_ideal) ** 2)

            if deviation < min_deviation:
                min_deviation = deviation
                best_function = column

        best_functions.append(best_function)

    return best_functions

best_functions = find_ideal_functions(engine)
print("Best fit ideal functions:", best_functions)



def match_test_data(engine, best_functions):
    test_data = pd.read_csv('test.csv')
    ideal_functions = pd.read_sql('ideal_functions', engine)

    results = []

    for _, row in test_data.iterrows():
        x, y = row['x'], row['y']
        deviations = {}

        for function in best_functions:
            y_ideal = ideal_functions.loc[ideal_functions['x'] == x, function].values[0]
            deviation = (y - y_ideal) ** 2
            deviations[function] = deviation

        best_fit_function = min(deviations, key=deviations.get)
        best_fit_deviation = deviations[best_fit_function]

        results.append([x, y, best_fit_function, best_fit_deviation])

    results_df = pd.DataFrame(results, columns=['x', 'y', 'ideal_function', 'deviation'])
    results_df.to_sql('test_data_results', engine, index=False, if_exists='replace')

    print("Test data matched to ideal functions and saved to the database.")

match_test_data(engine, best_functions)





def visualize_data(engine):
    training_data = pd.read_sql('training_data', engine)
    ideal_functions = pd.read_sql('ideal_functions', engine)
    test_data_results = pd.read_sql('test_data_results', engine)

    output_file("visualization.html")

    plots = []

    for i in range(1, 5):
        p = figure(title=f"Training Data vs Ideal Function {i}", x_axis_label='x', y_axis_label='y')

        p.scatter(training_data['x'], training_data[f'y{i}'], color='blue', legend_label='Training Data')

        ideal_function = best_functions[i-1]
        p.line(ideal_functions['x'], ideal_functions[ideal_function], color='green', legend_label='Ideal Function')

        test_data_filtered = test_data_results[test_data_results['ideal_function'] == ideal_function]
        p.scatter(test_data_filtered['x'], test_data_filtered['y'], color='red', legend_label='Test Data')

        plots.append(p)

    show(column(*plots))

visualize_data(engine)




class TestIdealFunctions(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine('sqlite:///ideal_functions.db')
        self.best_functions = find_ideal_functions(self.engine)

    def test_find_ideal_functions(self):
        self.assertEqual(len(self.best_functions), 4)

    def test_match_test_data(self):
        match_test_data(self.engine, self.best_functions)
        test_results = pd.read_sql('test_data_results', self.engine)
        self.assertFalse(test_results.empty)

if __name__ == '__main__':
    unittest.main()