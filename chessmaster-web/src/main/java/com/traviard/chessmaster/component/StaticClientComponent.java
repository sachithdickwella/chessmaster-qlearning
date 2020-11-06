package com.traviard.chessmaster.component;

import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/**
 * This {@link StaticClientComponent} to manage and manipulate Python (backend)
 * program input and output results.
 * <p>
 * This is a custom TCP server program to listen and communicate with the
 * PyTorch model by simply sending frames of the chessboard on each move and
 * listen for model responses to update the upstream components and UI.
 *
 * @author Sachith Dickwella
 */
@PropertySource("classpath:app-config.properties")
@Scope("prototype")
@Component
public class StaticClientComponent {

    /**
     * The host name of the server program.
     */
    @Value("${app.tcp.server.socket.host}")
    private String host;
    /**
     * The port number, or 0 to use a port number that is automatically allocated.
     */
    @Value("#{T(Integer).parseInt(${app.tcp.server.socket.port})}")
    private int port;
    /**
     * Default buffer size for input and output streams.
     */
    @Value("#{T(Integer).parseInt(${app.io.buffer.size})}")
    private int bufferSize;

    /**
     * Push downloaded image from UI to the backend Python program as a byte[] with
     * the UUID came from the UI.
     *
     * @param id          of the UI as a {@link java.util.UUID} instance.
     * @param imageStream object came from UI multipart upload.
     * @throws IOException if the downstream push fails.
     */
    public void push(@NotNull String id, @NotNull InputStream imageStream) throws IOException {
        try (Socket socket = new Socket(host, port);
             OutputStream outputStream = socket.getOutputStream();
             InputStream inputStream = socket.getInputStream()
        ) {
            /*
             * Write the uuid and image stream to the downstream program and flush.
             */
            outputStream.write(id.getBytes(StandardCharsets.UTF_8));
            outputStream.write(imageStream.readAllBytes());
            outputStream.flush();
            /*
             *
             */
            byte[] buffer = inputStream.readAllBytes();

            System.out.println(new String(buffer)); // TODO - Update the UI with results.
            System.out.println("Hello");
        }
    }
}
