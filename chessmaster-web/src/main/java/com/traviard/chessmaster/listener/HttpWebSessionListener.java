package com.traviard.chessmaster.listener;

import com.traviard.chessmaster.component.StaticClientComponent;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.servlet.annotation.WebListener;
import javax.servlet.http.HttpSessionEvent;
import javax.servlet.http.HttpSessionListener;
import java.util.ArrayList;
import java.util.List;

import static com.traviard.chessmaster.util.AppConstants.CREATE_SESSION;
import static com.traviard.chessmaster.util.AppConstants.INVALIDATE_SESSION;

/**
 * @author Sachith Dickwella
 */
@Configuration
@WebListener
public class HttpWebSessionListener extends EventListener implements HttpSessionListener {

    /**
     * Holds the active session ids in a {@link ArrayList} as a local member variable.
     */
    private final List<String> sessionIds = new ArrayList<>();

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
        var sessionId = event.getSession().getId();

        sessionIds.add(sessionId);
        send(sessionId, CREATE_SESSION);
    }

    /**
     * Notification that a session is about to be invalidated.
     * The default implementation is a NO-OP.
     *
     * @param event the notification event.
     */
    @Override
    public void sessionDestroyed(@NotNull HttpSessionEvent event) {
        var sessionId = event.getSession().getId();

        sessionIds.remove(sessionId);
        send(sessionId, INVALIDATE_SESSION);
    }

    /**
     * Get and bind the local-member variable {@link #sessionIds} to the application
     * context as a 'singleton' bean for controller usage.
     *
     * @return the instance of {@link #sessionIds} list.
     */
    @Bean(name = "sessionIds")
    public List<String> getSessionIds() {
        return this.sessionIds;
    }
}
