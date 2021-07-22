"""
This is a workaround to be able to "pass" the number of balancing steps to the kernels as an argument. It would probably
be beneficial to wrap GPFlow config instead, but I can't think of the current approach being detrimental in any way for
the time being.
"""
NUMBER_OF_BALANCING_STEPS = 10


def set_number_balancing_steps(n_balancing_steps):
    """
    Sets default number of balancing steps in the construction of the GP equivalent state-space model.
    This is used in particular for combination of kernels such as sum and product. In practice 10 is more than enough
    but it is possible to iterate further.
    """
    global NUMBER_OF_BALANCING_STEPS
    NUMBER_OF_BALANCING_STEPS = n_balancing_steps
