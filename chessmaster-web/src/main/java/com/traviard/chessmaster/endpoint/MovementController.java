package com.traviard.chessmaster.endpoint;

import com.traviard.chessmaster.component.StaticClientComponent;
import com.traviard.chessmaster.util.NextMove;
import org.apache.commons.lang3.StringUtils;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.simp.SimpMessageHeaderAccessor;
import org.springframework.messaging.simp.SimpMessageType;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.util.Arrays;
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
     * Instance of {@link StaticClientComponent} to access external server socket
     * to push content.
     */
    private final StaticClientComponent serverComponent;
    /**
     * Instance of {@link SimpMessagingTemplate} to send messages.
     */
    private final SimpMessagingTemplate template;

    /**
     * Single-arg constructor to initialize {@link #serverComponent} local member to work
     * with file push to Python model.
     *
     * @param serverComponent which inject by the Application Context.
     */
    @Autowired
    public MovementController(@NotNull StaticClientComponent serverComponent,
                              @NotNull SimpMessagingTemplate template) {
        this.serverComponent = serverComponent;
        this.template = template;
    }

    /**
     * Garb the uploaded file (image) and push through the configured TCP socket
     * to the python program.
     *
     * @param file instance of {@link MultipartFile} which uploaded.
     * @return instance of {@link ResponseEntity} to tell the file upload status.
     */
    @PostMapping(path = "/grab", consumes = MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Void> grabImage(@RequestParam("file") MultipartFile file,
                                          @NotNull HttpServletRequest request) {
        try {
            String id = Arrays.stream(request.getCookies())
                    .filter(cookie -> cookie.getName().equalsIgnoreCase("JSESSIONID"))
                    .map(Cookie::getValue)
                    .findAny()
                    .orElse(StringUtils.EMPTY);

            serverComponent.write(id, file.getInputStream());
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

    /**
     * Catch the next movement location of the chess picess from the downstream invoke
     * of this endpoint.
     *
     * @param nextMove of {@link NextMove} instance make the UI update.
     * @return instance of {@link ResponseEntity} to tell the next move submit status.
     */
    @PostMapping(path = "/next", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Void> nextMove(@RequestBody NextMove nextMove) {
        final String sessionId = nextMove.get_id();

        final SimpMessageHeaderAccessor accessor = SimpMessageHeaderAccessor
                .create(SimpMessageType.MESSAGE);
        accessor.setSessionId(sessionId);
        accessor.setLeaveMutable(true);

        template.convertAndSendToUser(sessionId,
                "/topic/next",
                nextMove,
                accessor.getMessageHeaders());

        LOGGER.info(nextMove.toString());
        return ResponseEntity.ok().build();
    }
}
