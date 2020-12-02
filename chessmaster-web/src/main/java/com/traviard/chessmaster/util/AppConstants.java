package com.traviard.chessmaster.util;

import org.jetbrains.annotations.NotNull;

/**
 * @author Sachith Dickwella
 */
public enum AppConstants {

    /**
     * Create session command message.
     */
    CREATE_SESSION("create"),
    /**
     * Invalidate session command message.
     */
    INVALIDATE_SESSION("invalidate"),
    /**
     * Clean all the sessions stored in the dictionary of Python program.
     */
    CLEAN_SESSIONS("clean"),
    /**
     * Splitter for two streams combined with {@code SequencedInputStream}.
     */
    SPLITTER("splitter");
    /**
     * Local variable to keep {@code constance} command or
     * {@link String} value.
     */
    private final String constant;

    /**
     * Private single-arg constructor to initialize {@link #constant}
     * local-member.
     *
     * @param constant to be initialize with enum value.
     */
    AppConstants(@NotNull String constant) {
        this.constant = constant;
    }

    /**
     * Get the initialized {@link #constant} value.
     *
     * @return the value of {@link #constant}.
     */
    @NotNull
    public String constant() {
        return this.constant;
    }
}
