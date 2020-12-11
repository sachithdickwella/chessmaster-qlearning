package com.traviard.chessmaster.util;

/**
 * @author Sachith Dickwella
 */
public enum SessionConstants {

    /**
     * FEN session attribute for the FEN string.
     */
    FEN("fen");

    /**
     * Actual session attribute name value.
     */
    private final String attribute;

    /**
     * Private constructor to initialize the member variable {@link #attribute}.
     *
     * @param attribute initial value for member {@link #attribute}.
     */
    SessionConstants(String attribute) {
        this.attribute = attribute;
    }

    /**
     * Get the {@link #attribute} member value instance for the enum instance.
     *
     * @return {@link #attribute} member for the enum.
     */
    public String attribute() {
        return this.attribute;
    }
}
