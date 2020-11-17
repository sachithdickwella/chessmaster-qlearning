package com.traviard.chessmaster.util;

import org.jetbrains.annotations.NotNull;

/**
 * Class to hold constant values for logging messages, so everytime when logging
 * happens, no need to initialize new {@link String}s.
 *
 * @author Sachith Dickwella
 */
public enum LogMessages {

    /**
     * When the file push to client queue is success.
     */
    INFO_FILE_PUSH_SUCCESS("{}: '{}' pushed to client queue - {} bytes"),
    /**
     * When the response from Python model has arrived.
     */
    INFO_RESPONSE_FROM_PYTHON_MODEL("Response from the Python model: {}"),
    /**
     * When the file push to client queue is fails.
     */
    ERROR_FILE_PUSH_FAILED("Uploaded file push is failed for id {} with java.io.IOException"),
    /**
     * When the command push to client queue is fails.
     */
    ERROR_COMMAND_PUSH_FAILED("Command push is failed for id {} with java.io.IOException");
    /**
     * Rendered message with {@link String#format(String, Object...)} in the
     * constructor.
     */
    private final String message;

    /**
     * Constructor to initialize enum with formatted message and its parameter
     * list to format.
     *
     * @param formattedMessage message with the format.
     * @param params           parameter list for formatted message.
     */
    LogMessages(@NotNull String formattedMessage, @NotNull Object... params) {
        this.message = format(formattedMessage, params);
    }

    /**
     * Get rendered {@link #message} from the enum.
     *
     * @return the formatted message from the enum.
     */
    public String message() {
        return this.message;
    }

    /**
     * Get rendered {@link #message} from the enum and perform post message
     * format with external parameters when return.
     *
     * @return the formatted message from the enum.
     */
    public String message(@NotNull Object... params) {
        return format(message(), params);
    }

    /**
     * Format the {@link String} with custom formatting pattern one by one with control
     * and return the rendered {@link String} value.
     *
     * @param message with formatting patterns.
     * @param params  to replace the patterns with actual values.
     * @return the formatted {@link String} value to the upstream.
     */
    private static String format(@NotNull String message, @NotNull Object... params) {
        for (Object param : params) {
            message = message.replaceFirst("\\{}", String.valueOf(param));
        }
        return message;
    }
}
