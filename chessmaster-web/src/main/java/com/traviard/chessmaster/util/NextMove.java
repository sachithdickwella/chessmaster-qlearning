package com.traviard.chessmaster.util;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.StringJoiner;

/**
 * @author Sachith Dickwella
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public record NextMove(String _id, String _wsid, String move) {

    @Override
    public String toString() {
        return new StringJoiner(", ", NextMove.class.getSimpleName() + "[", "]")
                .add("_id='" + _id + "'")
                .add("_wsid=" + _wsid + "'")
                .add("move='" + move + "'")
                .toString();
    }
}
