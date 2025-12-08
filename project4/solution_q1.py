from itertools import product
from typing import Any


class Factor:
    """This represents a factor in the Bayesian network, 
    - also represented as a conditional probability table 
    """

    def __init__(self, variables, probabilities):
        """
        Initialize a factor with its variables and probabilities
        Args:
            variables: list of variable names (e.g., ['B', 'E', 'A'])
            probabilities: dictionary of probability values
            - the dictionary maps the tuples of values to probability values
            - for example: {('+b', '+e', '+a'): 0.95, 0.002, etc}
        """
        self.variables = variables
        self.probabilities = probabilities

    def __repr__(self):
        return f"Factor({self.variables})"

def create_alarm_network_factors():
    """
    Create and return the factors for the alarm bayesiannetwork
    """

    #P(B) = prior probability of burglary
    factor_b = Factor(['B'], {'+b': 0.001, '-b': 0.999})

    #P(E) = prior probability of earthquake
    factor_e = Factor(['E'], {'+e': 0.002, '-e': 0.998})

    #P(A | B, E) = condional probability of alarm given burglary and earthquake
    factor_a = Factor(
        variables = ['B', 'E', 'A'], 
        probabilities={
            ('+b', '+e', '+a'): 0.95,
            ('+b', '+e', '-a'): 0.05,
            ('+b', '-e', '+a'): 0.94,
            ('+b', '-e', '-a'): 0.06,
            ('-b', '+e', '+a'): 0.29,
            ('-b', '+e', '-a'): 0.71,
            ('-b', '-e', '+a'): 0.001,
            ('-b', '-e', '-a'): 0.999
        }
    )

    #P(J|A) = conditional probability of John calling given alarm
    factor_j = Factor(
        variables = ['A', 'J'],
        probabilities={
            ('+a', '+j'): 0.9,
            ('+a', '-j'): 0.1,
            ('-a', '+j'): 0.05,
            ('-a', '-j'): 0.95
        }
    )

    #P(M|A) = conditional provavility of Mary calling given alarm
    factor_m = Factor(
        variables = ['A', 'M'],
        probabilities={
            ('+a', '+m'): 0.7,
            ('+a', '-m'): 0.3,
            ('-a', '+m'): 0.01,
            ('-a', '-m'): 0.99
        }
    )

    return [factor_b, factor_e, factor_a, factor_j, factor_m]

def restrict_factor(factor, variable, value):
    """
    restrict a factor by setting a variable equal to a specific value
    It returns a new factor with the variable removed 
    """

    #find the index of a variable to restrict
    if variable not in factor.variables:
        return factor
    
    var_index = factor.variables.index(variable)

    #create a new variables list withouth the restricted variable
    new_variables = []
    for i, v in enumerate(factor.variables):
        if i != var_index:
            new_variables.append(v)

    #create new probabilities by filtering entries matching the evidence
    new_probabilities = {}
    for assignment, prob in factor.probabilities.items():
        if assignment[var_index] == value:
            #Remove the restricted variable from the assignment tuple
            new_assignment_list = []
            for i in range(len(assignment)):
                if i != var_index:
                    new_assignment_list.append(assignment[i])
            new_assignment = tuple(new_assignment_list)

            new_probabilities[new_assignment] = prob
    
    return Factor(new_variables, new_probabilities)


def multiply_factors(factor1, factor2):
    """
    Multiply two factors together
    it returns a new factor with combined variables 
    """

    #First, find common and unique vairables 
    common_vars = []
    for v in factor1.variables:
        if v in factor2.variables:
            common_vars.append(v)


    all_vars = list(factor1.variables)
    extra_vars = []
    for v in factor2.variables:
        if v not in factor1.variables: 
            extra_vars.append(v)
    all_vars = all_vars + extra_vars

    # Create index mappings
    f1_indices = {}
    for v in factor1.variables:
        f1_indices[v] = factor1.variables.index(v)
    # Create index mappings
    f2_indices = {}
    for v in factor2.variables:
        f2_indices[v] = factor2.variables.index(v)
    
    new_probabilities = {} #build a new proabiltiy table

    # Generate all possible assignments for combined variables
    domains = {}
    for v in all_vars:
        pos = '+' + v.lower()
        neg = '-' + v.lower()
        domains[v] = [pos, neg]


    for assignment_tuple in product(*[domains[v] for v in all_vars]):
        #Extract assignments for each factor
        f1_assignment = tuple(assignment_tuple[all_vars.index(v)] for v in factor1.variables)
        f2_assignment = tuple(assignment_tuple[all_vars.index(v)] for v in factor2.variables)
        
        #Check if both assignments exist in their respective factors
        if f1_assignment in factor1.probabilities and f2_assignment in factor2.probabilities:
            new_probabilities[assignment_tuple] = (
                factor1.probabilities[f1_assignment] * factor2.probabilities[f2_assignment]
            )


def sum_out_variable(factor, variable):
    """
    sum out (marginalize) a variable from a factor
    It returns a new factor without that variable
    """

    if variable not in factor.variables:
        return factor
    
    var_index = factor.variables.index(variable)
    new_variables = []
    for v in factor.variables:
        if v != variable:
            new_variables.append(v)
    
    #grouping probabilities by assignments to other variables
    new_probabilities = {}

    for assignment, prob in factor.probabilities.items():
        #creae assignment without the summer out variable
        new_assignment_list = []
        for i in range(len(assignment)):
            if i != var_index:
                new_assignment_list.append(assignment[i])
        new_assignment = tuple(new_assignment_list)


        if new_assignment in new_probabilities:
            new_probabilities[new_assignment] += prob
        else:
            new_probabilities[new_assignment] = prob
    
    return Factor(new_variables, new_probabilities)


def main():
    """
    Main function used to compute P(Burglary | John Calls = +j)
    """

    # Create all factors for the Alarm network
    factors = create_alarm_network_factors()
    
    # Define the query: P(B | J = +j)
    query_variable = 'B'
    evidence = {'J': '+j'}
    
    # Define elimination order: eliminate E, A, M (not B or J)
    elimination_order = ['E', 'A', 'M']
    
    # Perform variable elimination
    result = variable_elimination(factors, query_variable, evidence, elimination_order)
    
    # Print the result
    print("P(Burglary | John Calls = +j):")
    print("=" * 40)
    
    # Sort by variable values for consistent output
    sorted_items = sorted(result.probabilities.items(), key=lambda x: x[0])
    
    for assignment, probability in sorted_items:
        var_value = assignment[0]
        if var_value == '+b':
            print(f"P(B = +b | J = +j) = {probability:.6f}")
        else:
            print(f"P(B = -b | J = +j) = {probability:.6f}")
    
    print("=" * 40)

if __name__ == "__main__":
    main()