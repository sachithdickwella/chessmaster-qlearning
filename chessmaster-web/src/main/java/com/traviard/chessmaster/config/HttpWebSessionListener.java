package com.traviard.chessmaster.config;

import com.traviard.chessmaster.component.StaticClientComponent;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;

import javax.servlet.annotation.WebListener;
import javax.servlet.http.HttpSessionEvent;
import javax.servlet.http.HttpSessionListener;

import static com.traviard.chessmaster.util.AppConstants.CREATE_SESSION;
import static com.traviard.chessmaster.util.AppConstants.INVALIDATE_SESSION;

/**
 * @author Sachith Dickwella
 */
@Configuration
@WebListener
public class HttpWebSessionListener extends EventListener implements HttpSessionListener {

    /**
     * Single-arg constructor to initialize {@code serverComponent} local member to work
     * with file push to Python model.
     *
     * @param serverComponent which inject by the Application Context.
     */
    @Autowired
    public HttpWebSessionListener(@NotNull StaticClientComponent serverComponent) {
        super(HttpWebSessionListener.class, serverComponent);
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
}
