def running_time_aspirate(amount):
    """Running time of aspirating in seconds.

    Parameters
    ----------
    amount : float
        The amount to aspirate in microliters.

    Returns
    -------
    running_time : float
        The running time in seconds.

    Notes
    -----
    Please note the default unit for the amount is in microliters.

    """
    running_time = 0.0109 * amount + 19.6445

    return running_time


def running_time_dispensing(amount):
    """Running time of dispensing in seconds.

    Parameters
    ----------
    amount : float
        The amount to dispense in microliters.

    Returns
    -------
    running_time : float
        The running time in seconds.

    Notes
    -----
    Please note the default unit for the amount is in microliters.

    """
    running_time = 0.0082 * amount + 16.1725

    return running_time


def running_time_washing(wash_volume, flush_volume):
    """Running time of washing single tips in seconds.

    Parameters
    ----------
    wash_volume : float
        The volume of wash solution in microliters.
    flush_volume : float
        The volume of flush solution in microliters.

    Returns
    -------
    running_time : float
        The running time in seconds.

    """
    # one wash cycle includes 1 wash and 1 flush
    running_time = 6.0270 * wash_volume + 32.0000 * flush_volume

    return running_time
