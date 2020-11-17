package com.traviard.chessmaster.component;

import com.traviard.chessmaster.util.AppConstants;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;

import javax.servlet.annotation.WebListener;
import javax.servlet.http.HttpSessionEvent;
import javax.servlet.http.HttpSessionListener;
import java.io.IOException;

import static com.traviard.chessmaster.util.AppConstants.CREATE_SESSION;
import static com.traviard.chessmaster.util.AppConstants.INVALIDATE_SESSION;
import static com.traviard.chessmaster.util.LogMessages.ERROR_COMMAND_PUSH_FAILED;

/**
 * @author Sachith Dickwella
 */
@Configuration
@WebListener
public class WebHttpSessionListener implements HttpSessionListener {

    /**
     * {@link Logger} for Spring-Boot compliant logging on console.
     */
    private static final Logger LOGGER = LoggerFactory.getLogger(WebHttpSessionListener.class);
    /**
     * Instance of {@link StaticClientComponent} to access external server socket
     * to push content.
     */
    private final StaticClientComponent serverComponent;

    /**
     * Single-arg constructor to initialize {@link #serverComponent} local member to work
     * with file push to Python model.
     *
     * @param serverComponent which inject by the Application Context.
     */
    @Autowired
    public WebHttpSessionListener(@NotNull StaticClientComponent serverComponent) {
        this.serverComponent = serverComponent;
    }

    /**
     * Notification that a session was created. The default implementation is a NO-OP.
     *
     * @param event the notification event.
     */
    @Override
    public void sessionCreated(@NotNull HttpSessionEvent event) {
        send(event.getSession().getId(), CREATE_SESSION);
    }

    /**
     * Notification that a session is about to be invalidated.
     * The default implementation is a NO-OP.
     *
     * @param event the notification event.
     */
    @Override
    public void sessionDestroyed(@NotNull HttpSessionEvent event) {
        send(event.getSession().getId(), INVALIDATE_SESSION);
    }

    /**
     * Common method to invoke on session creation and session invalidation.
     * On both operations, this method send out a command to the downstream
     * python program to prep the environment or clean the environment.
     *
     * @param id      from the session instance.
     * @param command to be executed on python program.
     */
    private void send(String id, AppConstants command) {
        try {
            serverComponent.write(id, command);
        } catch (IOException ex) {
            LOGGER.error(ERROR_COMMAND_PUSH_FAILED.message(id), ex);
        }
    }
}
