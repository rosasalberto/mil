class Error(Exception):
    """Base class for other exceptions"""
    pass

class DimensionError(Error):
    """Raised when the dimension is not the expected one"""
    pass 

class ExpectedListError(Error):
    """ Raised when the expected object has to be a list """
    pass
    
class FitNonCalledError(Error):
    """ Raised when fit has to be called before the current call """
    pass
    
class PrepareNonCalledError(Error):
    """ Raised when prepare has to be called before the current call """
    pass
    
class GetPositiveInstanceNotImplementedError(Error):
    """ Raised when the model has not implemented get_positive_instance_method """
    pass