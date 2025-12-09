from itertools import product
from typing import Any


class Factor:
    """
    This represents a factor in the Bayesian network, 
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
        #Display factor variables for debugging
        return f"Factor({self.variables})"

def create_alarm_network_factors():
    """
    Create and return the factors for the alarm bayesiannetwork
    """

    #P(B) = prior probability of burglary
    factor_b = Factor(['B'], {('+b',): 0.001, ('-b',): 0.999})

    #P(E) = prior probability of earthquake
    factor_e = Factor(['E'], {('+e',): 0.002, ('-e',): 0.998})

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

    #P(M|A) = conditional probability of Mary calling given alarm
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
    It returns a new factor with the variable removed, so any entries 
    that are inconsistent with the evidence are removed and variables are 
    dropped from the factor 
    """

    #find the index of a variable to restrict, and if If the factor does not involve this variable, return unchanged
    if variable not in factor.variables:
        return factor
    
    #index of the variable to eliminate
    var_index = factor.variables.index(variable)

    #create a new variables list without the restricted variable, 
    new_variables = [] #buulding a new variable list excluding the evidence variable
    for i, v in enumerate(factor.variables):
        if i != var_index:
            new_variables.append(v)

    #create new probabilities by filtering entries matching the evidence
    new_probabilities = {}
    for assignment, prob in factor.probabilities.items():
        if assignment[var_index] == value: #only keep rows where assignment[var] matches observed value
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

    - Variables are unified (all variables from both factors).
    - Only assignments consistent in shared variables are multiplied.
   
    """

    #First, find common and unique vairables 
    common_vars = []
    for v in factor1.variables:
        if v in factor2.variables:
            common_vars.append(v)

    #combine unique variables to form the new factor scope
    all_vars = list(factor1.variables)
    extra_vars = []
    for v in factor2.variables:
        if v not in factor1.variables: 
            extra_vars.append(v)
    all_vars = all_vars + extra_vars

    #Create index mapping, so like computing the index mappings for each factor 
    f1_indices = {}
    for v in factor1.variables:
        f1_indices[v] = factor1.variables.index(v)
    #Create index mappings
    f2_indices = {}
    for v in factor2.variables:
        f2_indices[v] = factor2.variables.index(v)
    
    new_probabilities = {} #build a new proabiltiy table

    #Generate all possible assignments for combined variables
    domains = {} #build domains for each variable ('+x', '-x')
    for v in all_vars:
        pos = '+' + v.lower()
        neg = '-' + v.lower()
        domains[v] = [pos, neg]

    #iterate over all the combinations of assignemnts for the unified scope
    for assignment_tuple in product(*[domains[v] for v in all_vars]):
        #Extract matchng assignments for each factor1 in correct order 
        f1_assignment = tuple(assignment_tuple[all_vars.index(v)] for v in factor1.variables)
        #Extract matching assignments for factor2
        f2_assignment = tuple(assignment_tuple[all_vars.index(v)] for v in factor2.variables)
        
        #Check if both assignments exist in their respective factors, and only multiply if it exists in the conditional probability tables
        if f1_assignment in factor1.probabilities and f2_assignment in factor2.probabilities:
            new_probabilities[assignment_tuple] = (
                factor1.probabilities[f1_assignment] * factor2.probabilities[f2_assignment]
            )

    return Factor(all_vars, new_probabilities)


def sum_out_variable(factor, variable):
    """
    sum out (marginalize) a variable from a factor
    It returns a new factor without that variable
    """

    #If variable not in this factor, nothing to do
    if variable not in factor.variables:
        return factor
    
    var_index = factor.variables.index(variable)
    new_variables = [] #Build new variable list without the eliminated variable
    for v in factor.variables:
        if v != variable:
            new_variables.append(v)
    
    #grouping probabilities by assignments to other variables, combining probabilities over eliminated variable
    new_probabilities = {}

    for assignment, prob in factor.probabilities.items():
        #create assignment without the eliminated variable
        new_assignment_list = []
        for i in range(len(assignment)):
            if i != var_index:
                new_assignment_list.append(assignment[i])
        new_assignment = tuple(new_assignment_list)

        #add probability into group bucket
        if new_assignment in new_probabilities:
            new_probabilities[new_assignment] += prob
        else:
            new_probabilities[new_assignment] = prob
    
    return Factor(new_variables, new_probabilities)


def normalize_factor(factor):
    """
    normalize a factor so probabilities sum to 1
    returns a new normalized factor
    """

    total = sum(factor.probabilities.values())
    
    if total == 0:
        return factor #avoiding dividing by 0
    
    new_probabilities = {
        assignment: prob/total
        for assignment, prob in factor.probabilities.items()
    }

    return Factor(factor.variables, new_probabilities)


def variable_elimination(factors, query_var, evidence, elimination_order):
    """
    Doing the variable elimination algorithm
    
    factors: list of Factor objects
    query_var: variable to query, for example 'b'
    evidence: dict of variable -> value (example, {'J': '+j'})
    elimination_order: list of variables to eliminate in order
    
    this returns the normalized Factor for the query variable
    """

    #Step 1: Restrict factors with evidence
    restricted_factors = []
    for factor in factors:
        restricted_factor = factor
        for var, val in evidence.items():
            restricted_factor = restrict_factor(restricted_factor, var, val)
        restricted_factors.append(restricted_factor)
    
    #step 2: Eliminate variables one by one in a specified order
    current_factors = restricted_factors

    for var_to_eliminate in elimination_order:
        #Find all factors that mention this variable, splitting factors into those mentioning var and those that don't
        factors_with_var = [f for f in current_factors if var_to_eliminate in f.variables]
        factors_without_var = [f for f in current_factors if var_to_eliminate not in f.variables]
        
        #multiply all factors containing the variable
        if len(factors_with_var) > 0:
            product_factor = factors_with_var[0]
            for f in factors_with_var[1:]:
                product_factor = multiply_factors(product_factor, f)
            
            #Sum out the variable
            summed_factor = sum_out_variable(product_factor, var_to_eliminate)
            
            #Update factor list
            current_factors = factors_without_var + [summed_factor]
        else:
            current_factors = factors_without_var
    
    
    #Step 3: Multiply remaining factors
    if len(current_factors) == 0:
        return None
    
    result_factor = current_factors[0]
    for f in current_factors[1:]:
        result_factor = multiply_factors(result_factor, f)
    
    #Step 4: Normalize the final result
    result_factor = normalize_factor(result_factor)
    
    return result_factor



def main():
    """
    Main function used to compute P(Burglary | John Calls = +j)
    """

    #Create all conditional probability table factors for the Alarm network
    factors = create_alarm_network_factors()
    
    #Define the query: P(B | J = +j) with a query variable
    query_variable = 'B'
    evidence = {'J': '+j'} #evidence that john calls = +j
    
    #Variables to eliminate, in this case, E, A, M -> based on the BN structure
    elimination_order = ['E', 'A', 'M']
    
    #do variable elimination
    result = variable_elimination(factors, query_variable, evidence, elimination_order)
    
    #Print the result
    print("P(Burglary | John Calls = +j):")
    print("=" * 40)
    
    #Sort so +b prints before -b consistently
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