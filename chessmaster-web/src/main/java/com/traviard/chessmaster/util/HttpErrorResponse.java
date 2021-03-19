package com.traviard.chessmaster.util;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDateTime;

/**
 * @author Sachith Dickwella
 */
@SuppressWarnings("unused")
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(ignoreUnknown = true)
public final class HttpErrorResponse {

    /**
     * Instance of {@link LocalDateTime} which error occurred.
     */
    private LocalDateTime timestamp;
    /**
     * HTTP status code for this error.
     */
    private Integer status;
    /**
     * HTTP status code message as a {@link String}.
     */
    private String error;
    /**
     * {@link String} message for the error instance.
     */
    private String message;
    /**
     * Endpoint for request which failed.
     */
    private String path;

    /**
     * Private constructor to avoid explicit object creation.
     */
    private HttpErrorResponse() {
        // do nothing.
    }

    /**
     * Initialize a new {@link Builder} instance and return.
     *
     * @return an instance of {@link Builder}.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder pattern implementation class which initialize all of its super
     * classes local members. In this case initialize local/instance members of
     * {@link HttpErrorResponse} class.
     */
    public static class Builder {

        /**
         * Instance to refer super class members which should pattern be applied.
         */
        @JsonIgnore
        private final HttpErrorResponse errorResponse;

        /**
         * Constructor to initialize {@link #errorResponse} instance during super
         * class's {@link HttpErrorResponse#builder()} invoke.
         */
        private Builder() {
            this.errorResponse = new HttpErrorResponse();
        }

        /**
         * Initialize {@link HttpErrorResponse#timestamp} with argument value.
         *
         * @param timestamp of the object creation.
         * @return this {@link Builder} instance to continue the build chain.
         */
        public Builder timestamp(LocalDateTime timestamp) {
            errorResponse.timestamp = timestamp;
            return this;
        }

        /**
         * Initialize {@link HttpErrorResponse#status} with argument value.
         *
         * @param status of the error created.
         * @return this {@link Builder} instance to continue the build chain.
         */
        public Builder status(Integer status) {
            errorResponse.status = status;
            return this;
        }

        /**
         * Initialize {@link HttpErrorResponse#error} with argument value.
         *
         * @param error of the object entitled to.
         * @return this {@link Builder} instance to continue the build chain.
         */
        public Builder error(String error) {
            errorResponse.error = error;
            return this;
        }

        /**
         * Initialize {@link HttpErrorResponse#message} with argument value.
         *
         * @param message of the error object.
         * @return this {@link Builder} instance to continue the build chain.
         */
        public Builder message(String message) {
            errorResponse.message = message;
            return this;
        }

        /**
         * Initialize {@link HttpErrorResponse#path} with argument value.
         *
         * @param path of the request targeted.
         * @return this {@link Builder} instance to continue the build chain.
         */
        public Builder path(String path) {
            errorResponse.path = path;
            return this;
        }

        /**
         * Finalize the builder chain by returning the {@link #errorResponse}
         * instance to the upstream.
         *
         * @return this instance of {@link #errorResponse} to upstream.
         */
        public HttpErrorResponse build() {
            return errorResponse;
        }
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public Integer getStatus() {
        return status;
    }

    public String getError() {
        return error;
    }

    public String getMessage() {
        return message;
    }

    public String getPath() {
        return path;
    }
}
