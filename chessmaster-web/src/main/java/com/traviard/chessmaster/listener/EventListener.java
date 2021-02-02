package com.traviard.chessmaster.listener;

import com.traviard.chessmaster.component.StaticClientWriterComponent;
import com.traviard.chessmaster.util.AppConstants;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import static com.traviard.chessmaster.util.LogMessages.ERROR_COMMAND_PUSH_FAILED;

/**
 * @author Sachith Dickwella
 */
public abstract class EventListener {

    /**
     * {@link Logger} for Spring-Boot compliant logging on console.
     */
    private final Logger logger;
    /**
     * Instance of {@link StaticClientWriterComponent} to access external server socket
     * to push content.
     */
    private final StaticClientWriterComponent serverComponent;

    /**
     * Protected constructor to initialize local members of {@link #logger} for logging
     * and {@link #serverComponent} for socket communication with downstream Python program.
     *
     * @param clazz           to be logged for the operations.
     * @param serverComponent to make socket communications with downstream programs.
     */
    protected EventListener(@NotNull Class<? extends EventListener> clazz,
                            @NotNull StaticClientWriterComponent serverComponent) {
        this.logger = LoggerFactory.getLogger(clazz);
        this.serverComponent = serverComponent;
    }

    /**
     * Common method to invoke on session creation and session invalidation.
     * On both operations, this method send out a command to the downstream
     * python program to prep the environment or clean the environment.
     *
     * @param id      from the session instance.
     * @param command to be executed on python program.
     */
    public void send(@NotNull String id, @NotNull AppConstants command) {
        try {
            serverComponent.write(id, command);
        } catch (IOException ex) {
            logger.error(ERROR_COMMAND_PUSH_FAILED.message(id), ex);
        }
    }
}
