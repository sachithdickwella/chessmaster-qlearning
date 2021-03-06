package com.traviard.chessmaster.endpoint;

import com.traviard.chessmaster.component.StaticClientWriterComponent;
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
import javax.servlet.http.HttpSession;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.SequenceInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static com.traviard.chessmaster.util.AppConstants.*;
import static com.traviard.chessmaster.util.LogMessages.*;
import static com.traviard.chessmaster.util.SessionConstants.FEN;
import static com.traviard.chessmaster.util.SessionConstants.IS_TRAIN;
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
     * Instance of {@link StaticClientWriterComponent} to access external server socket
     * to push content.
     */
    private final StaticClientWriterComponent serverComponent;
    /**
     * Instance of {@link SimpMessagingTemplate} to send messages.
     */
    private final SimpMessagingTemplate template;

    /**
     * Single-arg constructor to initialize {@link #serverComponent} local member
     * to work with file push to Python model.
     *
     * @param serverComponent which inject by the Application Context.
     * @param template        to send WebSocket responses initiate by the server.
     */
    @Autowired
    public MovementController(@NotNull StaticClientWriterComponent serverComponent,
                              @NotNull SimpMessagingTemplate template) {
        this.serverComponent = serverComponent;
        this.template = template;
    }

    /**
     * Garb the uploaded file (image) and push through the configured TCP socket
     * to the python program.
     *
     * @param frame1  instance of {@link MultipartFile} which uploaded previous
     *                status of the chessboard.
     * @param frame2  instance of {@link MultipartFile} which uploaded new status
     *                of the chessboard.
     * @param fen     Forsyth–Edwards Notation (FEN) string to keep in the session on
     *                every move from front-end.
     * @param request {@link HttpServletRequest} instance for the current request.
     * @return instance of {@link ResponseEntity} to tell the file upload status.
     */
    @SuppressWarnings({"java:S2629", "java:S5411"})
    @PostMapping(path = "/grab", consumes = MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Void> grabImage(@RequestParam("frame1") MultipartFile frame1,
                                          @RequestParam("frame2") MultipartFile frame2,
                                          @RequestParam("fen") String fen,
                                          @RequestParam("isTrain") Boolean isTrain,
                                          @NotNull HttpServletRequest request) {

        var cookies = request.getCookies();

        var id = Arrays.stream(cookies)
                .filter(cookie -> cookie.getName().equalsIgnoreCase("JSESSIONID"))
                .map(Cookie::getValue)
                .findAny()
                .orElse(StringUtils.EMPTY);

        var wsid = Arrays.stream(cookies)
                .filter(cookie -> cookie.getName().equalsIgnoreCase("SID"))
                .map(Cookie::getValue)
                .findAny()
                .orElse(StringUtils.EMPTY);

        try {
            var imageSequence = new SequenceInputStream(Collections.enumeration(List.of(
                    frame1.getInputStream(),
                    new ByteArrayInputStream(SPLITTER.constant().getBytes(StandardCharsets.UTF_8)),
                    frame2.getInputStream()
            )));

            serverComponent.write(id, wsid, isTrain, imageSequence);

            LOGGER.info(INFO_FILE_PUSH_SUCCESS.message(
                    id,
                    new StringBuilder(3)
                            .append(Optional.ofNullable(frame1.getOriginalFilename()).orElse("<NoFileName>"))
                            .append(" and ")
                            .append(Optional.ofNullable(frame2.getOriginalFilename()).orElse("<NoFileName>")),
                    frame1.getSize() + frame2.getSize()));

            HttpSession session = request.getSession();
            session.setAttribute(FEN.attribute(), fen);
            session.setAttribute(IS_TRAIN.attribute(), isTrain);

            return ResponseEntity.ok().build();
        } catch (IOException ex) {
            LOGGER.error(ERROR_FILE_PUSH_FAILED.message(id), ex);
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).build();
        }
    }

    /**
     * Catch the next movement location of the chess pieces from the downstream invoke
     * of this endpoint.
     *
     * @param nextMove of {@link NextMove} instance make the UI update.
     * @return instance of {@link ResponseEntity} to tell the next move submit status.
     */
    @SuppressWarnings("java:S2629")
    @PostMapping(path = "/next", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Void> nextMove(@RequestBody NextMove nextMove) {
        LOGGER.info(INFO_RESPONSE_FROM_PYTHON_MODEL.message(nextMove));

        final String sessionId = nextMove._wsid();

        final SimpMessageHeaderAccessor accessor = SimpMessageHeaderAccessor
                .create(SimpMessageType.MESSAGE);
        accessor.setSessionId(sessionId);
        accessor.setLeaveMutable(true);

        template.convertAndSendToUser(sessionId,
                "/queue/next",
                nextMove,
                accessor.getMessageHeaders());

        return ResponseEntity.ok().build();
    }
}
