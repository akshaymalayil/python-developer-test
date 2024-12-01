import numpy as np
import matplotlib.pyplot as plt


def calculate_probabilities(parameters, data, utilities):
    """
    Calculate probabilities for a multinomial logit model.

    Parameters:
        parameters (dict): Dictionary containing the β coefficients.
        data (dict): Dictionary containing the independent variables (X1, X2, Sero, etc.).
        utilities (list): List of functions defining the deterministic utilities.

    Returns:
        dict: Dictionary with keys representing each alternative and values as lists
              containing the calculated probabilities for each data point.
    """
    # Validate inputs
    try:
        # Ensure all data lists are of the same length
        n_points = len(next(iter(data.values())))
        for key, values in data.items():
            if len(values) != n_points:
                raise ValueError(f"Data length mismatch for variable '{key}'. Expected {n_points}, got {len(values)}.")

        # Initialize utilities and probabilities
        utility_values = {f'V{i+1}': [] for i in range(len(utilities))}
        probabilities = {f'P{i+1}': [] for i in range(len(utilities))}

        # Calculate utilities for each alternative
        for i in range(n_points):
            data_point = {key: values[i] for key, values in data.items()}
            for idx, utility_func in enumerate(utilities):
                utility_values[f'V{idx+1}'].append(utility_func(parameters, data_point))

        # Compute probabilities
        for i in range(n_points):
            exp_values = [np.exp(utility_values[f'V{j+1}'][i]) for j in range(len(utilities))]
            sum_exp = sum(exp_values)
            for idx, exp_value in enumerate(exp_values):
                probabilities[f'P{idx+1}'].append(exp_value / sum_exp)

        # Validate probabilities
        for i in range(n_points):
            total = sum(probabilities[f'P{j+1}'][i] for j in range(len(utilities)))
            assert abs(total - 1) < 1e-6, f"Probabilities do not sum to 1 for data point {i}."

        return probabilities

    except Exception as e:
        print(f"Error: {e}")
        return None


def visualize_probabilities(probabilities):
    """
    Visualize the probabilities for each alternative.

    Parameters:
        probabilities (dict): Dictionary containing probabilities for each alternative.
    """
    try:
        # Data points
        data_points = range(len(probabilities['P1']))

        # Plot probabilities
        plt.figure(figsize=(10, 6))
        plt.plot(data_points, probabilities['P1'], label='P1', marker='o')
        plt.plot(data_points, probabilities['P2'], label='P2', marker='s')
        plt.plot(data_points, probabilities['P3'], label='P3', marker='^')

        # Customize the plot
        plt.title('Probabilities for Each Alternative', fontsize=14)
        plt.xlabel('Data Point Index', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)

        # Save and show the plot
        plt.savefig("probability_visualization.png")
        plt.show()
        print("Visualization saved as 'probability_visualization.png'.")

    except Exception as e:
        print(f"Visualization Error: {e}")


if __name__ == "__main__":
    # Example data and parameters
    parameters = {'β01': 0.1, 'β1': 0.5, 'β2': 0.5, 'β02': 1, 'β03': 0, 'β4': 0.2}
    data = {
        'X1': [2, 3, 5, 7, 1, 8, 4, 5, 6, 7],
        'X2': [1, 5, 3, 8, 2, 7, 5, 9, 4, 2],
        'Sero': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    # Utility functions
    utilities = [
        lambda params, d: params['β01'] + params['β1'] * d['X1'] + params['β2'] * d['X2'] + params['β4'] * d['X2'],
        lambda params, d: params['β02'] + params['β1'] * d['X1'] + params['β2'] * d['X2'],
        lambda params, d: params['β03'] + params['β1'] * d['Sero'] + params['β2'] * d['Sero'],
    ]

    # Calculate probabilities
    probabilities = calculate_probabilities(parameters, data, utilities)

    # Save output to file in the same folder
    if probabilities:
        output_file = "output.txt"
        try:
            with open(output_file, 'w') as file:
                file.write("Calculated Probabilities:\n")
                file.write(str(probabilities))
            print(f"Probabilities saved to '{output_file}'.")

            # Visualize probabilities
            visualize_probabilities(probabilities)

        except Exception as e:
            print(f"Failed to save output: {e}")
