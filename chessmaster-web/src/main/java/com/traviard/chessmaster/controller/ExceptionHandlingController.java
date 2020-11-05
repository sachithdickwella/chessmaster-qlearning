package com.traviard.chessmaster.controller;

import com.traviard.chessmaster.util.HttpErrorResponse;
import org.jetbrains.annotations.NotNull;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

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
}
