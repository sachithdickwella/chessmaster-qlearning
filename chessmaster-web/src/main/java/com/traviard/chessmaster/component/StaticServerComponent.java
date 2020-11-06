package com.traviard.chessmaster.component;

import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.IOException;
import java.io.InputStream;
import java.net.ServerSocket;

import static com.traviard.chessmaster.util.LogMessages.INFO_STATIC_SERVER_INIT_SUCCESS;

/**
 * This {@link StaticServerComponent} to manage and manipulate Python (backend)
 * program input and output results.
 * <p>
 * This is a custom TCP server program to listen and communicate with the
 * PyTorch model by simply sending frames of the chessboard on each move and
 * listen for model responses to update the upstream components and UI.
 *
 * @author Sachith Dickwella
 */
@PropertySource("classpath:app-config.properties")
@Scope("singleton")
@Component
public class StaticServerComponent {

    /**
     * {@link Logger} for Spring-Boot compliant logging on console.
     */
    private static final Logger LOGGER = LoggerFactory.getLogger(StaticServerComponent.class);
    /**
     * The port number, or 0 to use a port number that is automatically allocated.
     */
    @Value("#{T(Integer).parseInt('${app.tcp.server.socket.port}')}")
    private int port;
    /**
     * Requested maximum length of the queue of incoming connections.
     */
    @Value("#{T(Integer).parseInt('${app.tcp.server.socket.backlog}')}")
    private int backlog;
    /**
     * Default buffer size for input and output streams.
     */
    @Value("#{T(Integer).parseInt('${app.io.buffer.size}')}")
    private int bufferSize;
    /**
     * Server socket to keep open a channel with backend.
     */
    private ServerSocket serverSocket;

    /**
     * Post initialize the {@link StaticServerComponent}' {@link ServerSocket} with
     * configured {@link #port} and {@link #backlog} values.
     *
     * @throws IOException when the {@link #serverSocket} initialization fails.
     */
    @PostConstruct
    public void init() throws IOException {
        serverSocket = new ServerSocket(port, backlog);
        LOGGER.info(INFO_STATIC_SERVER_INIT_SUCCESS.message(port));
    }

    /**
     * Close the {@link #serverSocket} instance just before the component object be
     * destroyed.
     *
     * @throws IOException when the {@link #serverSocket} initialization fails.
     */
    @PreDestroy
    public void destroy() throws IOException {
        serverSocket.close();
    }

    /**
     *
     */
    public void push(@NotNull String id, @NotNull InputStream inputStream) throws IOException {
    }

    /**
     *
     */
    private void listen() {
    }
}
