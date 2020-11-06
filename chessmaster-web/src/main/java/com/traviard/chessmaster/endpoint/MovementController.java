package com.traviard.chessmaster.endpoint;

import com.traviard.chessmaster.component.StaticServerComponent;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Optional;

import static com.traviard.chessmaster.util.LogMessages.INFO_FILE_PUSH_FAILED;
import static com.traviard.chessmaster.util.LogMessages.INFO_FILE_PUSH_SUCCESS;
import static org.springframework.http.MediaType.MULTIPART_FORM_DATA_VALUE;

/**
 * This controller grab the movement changed screenshot of the front-end
 * chess board and push it the Q-Learning model via TCP socket.
 *
 * @author Sachith Dickwella
 */
@RestController
@RequestMapping(path = "/movement")
public class MovementController {

    /**
     * {@link Logger} for Spring-Boot compliant logging on console.
     */
    private static final Logger LOGGER = LoggerFactory.getLogger(MovementController.class);
    /**
     * Instance of {@link StaticServerComponent} to access external server socket
     * to push content.
     */
    private final StaticServerComponent serverComponent;

    /**
     * Single-arg constructor to initialize {@link #serverComponent} local member to work
     * with file push to Python model.
     *
     * @param serverComponent which inject by the Application Context.
     */
    @Autowired
    public MovementController(@NotNull StaticServerComponent serverComponent) {
        this.serverComponent = serverComponent;
    }

    /**
     * Garb the uploaded file (image) and push through the configured TCP socket
     * to the python program.
     *
     * @param file instance of {@link MultipartFile} which uploaded.
     * @return instance of {@link ResponseEntity} to tell the file upload status.
     */
    @PostMapping(path = "/grab", consumes = MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Void> grabImage(@RequestParam("id") String id, @RequestParam("file") MultipartFile file) {
        try {
            serverComponent.push(id, file.getInputStream());
            LOGGER.info(INFO_FILE_PUSH_SUCCESS.message(
                    id,
                    Optional.ofNullable(file.getOriginalFilename()).orElse("<NoFileName>"),
                    file.getBytes().length));

            return ResponseEntity.ok().build();
        } catch (IOException ex) {
            LOGGER.error(INFO_FILE_PUSH_FAILED.message(), ex);
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).build();
        }
    }
}
