package com.traviard.chessmaster.component;

import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
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
     * Push the downloaded image from UI to the backend Python program as a byte[] with
     * the UUID came from the UI.
     *
     * @param id          of the UI as a {@link java.util.UUID} instance.
     * @param imageStream object came from UI multipart upload.
     * @throws IOException if the downstream push fails.
     */
    public void write(@NotNull String id, @NotNull InputStream imageStream) throws IOException {
        try (SocketChannel channel = SocketChannel.open(new InetSocketAddress(host, port));
             var sequenceStream = new SequenceInputStream(Collections.enumeration(
                     List.of(
                             new ByteArrayInputStream(id.getBytes(StandardCharsets.UTF_8)),
                             imageStream
                     ))
             )) {
            /*
             * Write the uuid and image stream to the downstream program and flush.
             */
            channel.write(ByteBuffer.wrap(sequenceStream.readAllBytes()));

            Socket s = channel.socket();
            int count = s.getInputStream().read();

            /*ByteBuffer buffer = ByteBuffer.allocate(256);
            channel.read(buffer);*/

            System.out.println(count);
        }
    }
}
