package com.traviard.chessmaster.controller;

import com.traviard.chessmaster.util.HttpErrorResponse;
import org.jetbrains.annotations.NotNull;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;

import javax.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.time.LocalDateTime;

/**
 * @author Sachith Dickwella
 */
@ControllerAdvice
public class ExceptionHandlingController {

    /**
     * Send out BAD REQUEST (HTTP 400) error when the uploaded file or any other
     * IO operation fails.
     *
     * @param request of {@link HttpServletRequest} ingested from servlet container.
     * @return an instance of {@link ResponseEntity} of {@link String}.
     */
    @ExceptionHandler(IOException.class)
    public ResponseEntity<HttpErrorResponse> common(@NotNull HttpServletRequest request) {
        return ResponseEntity.badRequest()
                .body(HttpErrorResponse.builder()
                        .timestamp(LocalDateTime.now())
                        .status(HttpStatus.BAD_REQUEST.value())
                        .error(HttpStatus.BAD_REQUEST.getReasonPhrase())
                        .message("Problem with a I/O operations in the path")
                        .path(request.getRequestURI())
                        .build());
    }

    /**
     * Send out BAD REQUEST (HTTP 400) error when the uploaded file exceeds it configured limit.
     *
     * @param request of {@link HttpServletRequest} ingested from servlet container.
     * @return an instance of {@link ResponseEntity} of {@link String}.
     */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<HttpErrorResponse> maximumFileSizeExceeded(@NotNull HttpServletRequest request) {
        return ResponseEntity.badRequest()
                .body(HttpErrorResponse.builder()
                        .timestamp(LocalDateTime.now())
                        .status(HttpStatus.BAD_REQUEST.value())
                        .error(HttpStatus.BAD_REQUEST.getReasonPhrase())
                        .message("Maximum file size exceeded. File size should be 4MB or less.")
                        .path(request.getRequestURI())
                        .build());
    }

    /**
     * Send out BAD REQUEST (HTTP 400) error, servlet request part is missing in request form.
     *
     * @param request of {@link HttpServletRequest} ingested from servlet container.
     * @param ex      of the encountered {@link MissingServletRequestPartException} object.
     * @return an instance of {@link ResponseEntity} of {@link String}.
     */
    @ExceptionHandler(MissingServletRequestPartException.class)
    public ResponseEntity<HttpErrorResponse> missingRequestPart(@NotNull HttpServletRequest request,
                                                                @NotNull MissingServletRequestPartException ex) {
        return ResponseEntity.badRequest()
                .body(HttpErrorResponse.builder()
                        .timestamp(LocalDateTime.now())
                        .status(HttpStatus.BAD_REQUEST.value())
                        .error(HttpStatus.BAD_REQUEST.getReasonPhrase())
                        .message(ex.getMessage())
                        .path(request.getRequestURI())
                        .build());
    }
}
