package com.traviard.chessmaster.endpoint;

import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * @author Sachith Dickwella
 */
@RestController
@RequestMapping(path = "/session")
public class SessionManagementController {

    /**
     * {@link List} of {@link String} session ids from the application context.
     */
    private final List<String> sessionIds;

    /**
     * Single-arg constructor to initialize local member {@link #sessionIds} from
     * the application context by autowiring.
     *
     * @param sessionIds {@link List} from the application context to be used
     *                   in the controller.
     */
    @Autowired
    public SessionManagementController(@NotNull List<String> sessionIds) {
        this.sessionIds = sessionIds;
    }

    /**
     * Request mapping to retrieve {@link #sessionIds} local instance as a JSON
     * array for the invoker.
     *
     * @return an instance of {@link ResponseEntity<List>} with currently active
     * session ids.
     */
    @GetMapping(value = "/ids", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<List<String>> sessionIds() {
        return ResponseEntity.ok(this.sessionIds);
    }
}
