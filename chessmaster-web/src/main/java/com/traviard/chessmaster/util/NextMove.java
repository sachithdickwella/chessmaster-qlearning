package com.traviard.chessmaster.util;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.util.StringJoiner;

/**
 * @author Sachith Dickwella
 */
@Data
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class NextMove {
    /**
     * UUID of the image originally
     * send the frame to the model.
     */
    @SuppressWarnings("java:S116")
    private String _id;
    /**
     * Response from the model that
     * states how the UI pieces should
     * be moved.
     */
    private String move;

    @Override
    public String toString() {
        return new StringJoiner(", ", NextMove.class.getSimpleName() + "[", "]")
                .add("uuid='" + _id + "'")
                .add("move='" + move + "'")
                .toString();
    }
}
