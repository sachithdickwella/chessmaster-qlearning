package com.traviard.chessmaster.util;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

/**
 * @author Sachith Dickwella
 */
@Getter
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(ignoreUnknown = true)
public class HttpErrorResponse {

    /**
     * Instance of {@link LocalDateTime} which error occurred.
     */
    private final LocalDateTime timestamp;
    /**
     * HTTP status code for this error.
     */
    private final int status;
    /**
     * HTTP status code message as a {@link String}.
     */
    private final String error;
    /**
     * {@link String} message for the error instance.
     */
    private final String message;
    /**
     * Endpoint for request which failed.
     */
    private final String path;
}
