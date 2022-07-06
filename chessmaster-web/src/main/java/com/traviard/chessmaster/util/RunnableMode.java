package com.traviard.chessmaster.util;

import org.jetbrains.annotations.NotNull;

import static java.util.Locale.ROOT;

/**
 * @author Sachith Dickwella
 */
public enum RunnableMode {

    /**
     * Application runs on training mode to deliver new data.
     */
    TRAIN("train"),
    /**
     * Application runs on evaluation mode to show the results.
     */
    EVAL("eval");

    /**
     * Local member to hold the mode value as a {@link String}.
     */
    private final String mode;

    /**
     * Private single-arg constructor to initialize each enum
     * with the mode {@link String} value.
     *
     * @param mode value to initialize {@link #mode} variable.
     */
    RunnableMode(@NotNull String mode) {
        this.mode = mode;
    }

    /**
     * Get the {@link #mode} value from each of the enum.
     *
     * @return the value of {@link #mode} reference.
     */
    @NotNull
    public String mode() {
        return this.mode;
    }

    /**
     * Get the {@link RunnableMode} value from the {@link String} mode
     * value from the parameter.
     *
     * @param mode value as a {@link String}.
     * @return an instance of {@link RunnableMode} for the parameter mode.
     */
    public static RunnableMode of(@NotNull String mode) {
        return valueOf(mode.toUpperCase(ROOT));
    }
}
