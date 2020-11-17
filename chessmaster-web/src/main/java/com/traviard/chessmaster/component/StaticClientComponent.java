package com.traviard.chessmaster.component;

import com.traviard.chessmaster.util.AppConstants;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;

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
     * Send out the command along with the {@code id} to the downstream Python program,
     * so the program would be able to capture and execute whatever command send out
     * before the actual objective of the program is required or after the objective of
     * the program is done.
     *
     * @param id      of the UI as a {@link String} instance.
     * @param command to be executed on downstream program from {@link AppConstants}.
     */
    public void write(@NotNull String id, @NotNull AppConstants command) throws IOException {
        try (var channel = SocketChannel.open(new InetSocketAddress(host, port));
             var sequenceStream = new SequenceInputStream(Collections.enumeration(
                     List.of(
                             new ByteArrayInputStream(id.getBytes(StandardCharsets.UTF_8)),
                             new ByteArrayInputStream(command.constant().getBytes(StandardCharsets.UTF_8))
                     ))
             )) {
            /*
             * Write the id and the command stream to the downstream program and flush.
             */
            channel.write(ByteBuffer.wrap(sequenceStream.readAllBytes()));
        }
    }

    /**
     * Push the downloaded image from UI to the backend Python program as a byte[] with
     * the UUID came from the UI.
     *
     * @param id          of the UI as a {@link String} instance.
     * @param wsid        which contains the websocket session id as a {@link String}.
     * @param imageStream object came from UI multipart upload.
     * @throws IOException if the downstream push fails.
     */
    public void write(@NotNull String id, @NotNull String wsid, @NotNull InputStream imageStream) throws IOException {
        try (var channel = SocketChannel.open(new InetSocketAddress(host, port));
             var sequenceStream = new SequenceInputStream(Collections.enumeration(
                     List.of(
                             new ByteArrayInputStream(id.getBytes(StandardCharsets.UTF_8)),
                             new ByteArrayInputStream(wsid.getBytes(StandardCharsets.UTF_8)),
                             imageStream
                     ))
             )) {
            /*
             * Write the id and image stream to the downstream program and flush.
             */
            channel.write(ByteBuffer.wrap(sequenceStream.readAllBytes()));
        }
    }
}
