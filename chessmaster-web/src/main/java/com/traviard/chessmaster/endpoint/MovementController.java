package com.traviard.chessmaster.endpoint;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

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
     * Garb the uploaded file (image) and push through the configured TCP socket
     * to the python program.
     *
     * @param file instance of {@link MultipartFile} which uploaded.
     * @return instance of {@link ResponseEntity} to tell the file upload status.
     */
    @PostMapping(path = "/grab", consumes = MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Void> grabImage(@RequestParam("file") MultipartFile file) {
        return ResponseEntity.ok().build();
    }
}
