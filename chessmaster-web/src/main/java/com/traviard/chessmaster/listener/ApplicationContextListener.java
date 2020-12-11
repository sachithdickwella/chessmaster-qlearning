package com.traviard.chessmaster.listener;

import com.traviard.chessmaster.component.StaticClientComponent;
import org.apache.commons.lang3.StringUtils;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;

import javax.servlet.ServletContextEvent;
import javax.servlet.ServletContextListener;
import javax.servlet.annotation.WebListener;

import static com.traviard.chessmaster.util.AppConstants.CLEAN_SESSIONS;

/**
 * @author Sachith Dickwella
 */
@Configuration
@WebListener
public class ApplicationContextListener extends EventListener implements ServletContextListener {

    /**
     * Single-arg constructor to initialize {@code serverComponent} local member to work
     * with file push to Python model.
     *
     * @param serverComponent which inject by the Application Context.
     */
    @Autowired
    public ApplicationContextListener(@NotNull StaticClientComponent serverComponent) {
        super(ApplicationContextListener.class, serverComponent);
    }

    /**
     * * Notification that the servlet context is about to be shut down. All
     * servlets and filters have been destroyed before any
     * ServletContextListeners are notified of context destruction.
     * The default implementation is a NO-OP.
     *
     * @param sce Information about the ServletContext that was destroyed.
     */
    @Override
    public void contextDestroyed(ServletContextEvent sce) {
        send(StringUtils.EMPTY, CLEAN_SESSIONS);
    }
}
