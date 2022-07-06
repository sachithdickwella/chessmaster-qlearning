# -*- coding: utf-8 -*-

class PromotionInvalidException(Exception):
    """
    Exception to raise when pawn promotion happens, and promotion value is not provided.
    """

    def __init__(self, message):
        super(PromotionInvalidException, self).__init__(message)
