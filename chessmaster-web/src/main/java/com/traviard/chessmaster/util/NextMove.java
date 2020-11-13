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
     * UUID of the image originally send the frame to the model.
     */
    @SuppressWarnings("java:S116")
    private String _id;
    /**
     * Target web socket session id which the model response should
     * be sent up to.
     */
    @SuppressWarnings("java:S116")
    private String _wsid;
    /**
     * Response from the model that states how the UI pieces should
     * be moved.
     */
    private String move;

    @Override
    public String toString() {
        return new StringJoiner(", ", NextMove.class.getSimpleName() + "[", "]")
                .add("_id='" + _id + "'")
                .add("_wsid=" + _wsid + "'")
                .add("move='" + move + "'")
                .toString();
    }
}
