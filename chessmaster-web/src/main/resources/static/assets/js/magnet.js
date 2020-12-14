let board, game, startFrame, _fen, _pgn;
/**
 * Connect and subscribe a WebSocket. in order to receive server responses from
 * the model.
 */
const connect = () => {
    const socket = new SockJS('/gs-guide-websocket');
    const stompClient = Stomp.over(socket);
    stompClient.connect({}, _ => {
        const sessionId = socket._transport.url
            .match('(ws|wss):\\/\\/\\w+(?::[0-9]+)?\\/\\S+\\/[0-9]+\\/(\\w+)\\/websocket')[2]

        document.cookie = `SID=${sessionId}; SameSite=strict; path=/`
        stompClient.subscribe('/user/queue/next', nextMove);
    });
};
/**
 * Make the move upon the server response from the model.
 *
 * @param next to give in the next move location on the board.
 */
const nextMove = (next) => {
    console.info(JSON.parse(next.body));
    // TODO - board.move(`b8-c6`); - move string should come from websocket: QL model.

    updateStats(/* This should update the color value opponent belongs to*/);
    _wait('auto');
};
/**
 * The event of new move by the user or the computer. Oftentimes, computer doesn't trigger
 * this event.
 *
 * @param source of the moved chess piece.
 * @param target or destination of the moved chess piece.
 * @param piece name that being moved.
 * @param newPos FEN (Forsyth–Edwards Notation) string for the new position of the board.
 * @param oldPos FEN (Forsyth–Edwards Notation) string for the old position of the board.
 * @param orientation of the board (white/black at below).
 */
const onDrop = (source, target, piece, newPos, oldPos, orientation) => {
    const moves = game.move({
        from: source,
        to: target,
        promotion: 'q' // NOTE: always promote to a queen for example simplicity
    })

    // Illegal move
    if (moves === null) return 'snapback'

    _wait('wait', 'block');

    setTimeout(() => {
        if (source !== target) shot(push);
    }, 500);
};
/**
 * Update the board position after the piece snap for castling,
 * en passant, pawn promotion.
 */
const onSnapEnd = () => board.position(game.fen())
/**
 * The event of new move startup by the user. Remove the square highlights and take a snapshot.
 *
 * @param source also know as square id of move.
 * @param piece which started to move.
 */
const onDragStart = (source, piece) => {
    if (game.game_over()) return false;

    if ((game.turn() === 'w' && piece.search(/^b/) !== -1)
        || (game.turn() === 'b' && piece.search(/^w/) !== -1)) return false;

    shot(retainBlob);
};
/**
 * Shoot the chessboard HTML as a canvas and push it to the server end as an image.
 *
 * @param callback of the shot function after successful snapshot.
 */
const shot = (callback) => html2canvas(document.querySelector("#board1")).then(canvas => callback(canvas));
/**
 * Retain the capturing blob in the 'startFrame' so, we can later use it.
 *
 * @param canvas of the shot out chessboard.
 */
const retainBlob = (canvas) => canvas.toBlob((blob) => startFrame = blob);
/**
 * Send out the shot out canvas to the backend server as an image/png via AJAX.
 *
 * @param canvas of the shot out chessboard.
 */
const push = (canvas) => canvas.toBlob((blob) => {
    const form = new FormData();
    form.append('frame1', startFrame, 'board-frame1');
    form.append('frame2', blob, 'board-frame2');
    form.append('fen', game.fen());
    $.ajax({
        url: '/movement/grab',
        method: 'POST',
        enctype: 'multipart/form-data',
        contentType: false,
        processData: false,
        cache: false,
        data: form,
        error: (jqXHR) => {
            console.log(jqXHR);
        }
    });
}, 'image/png');
/**
 * Update the status from the board status. Thi will provide a feedback to the
 * user to continue the game.
 *
 * @param t to update the game.turn if defined.
 */
const updateStats = (t) => {
    let status;
    if (t !== undefined && t !== '') game.turn(t);

    if (game.in_checkmate()) {
        status = `Game over, ${_longTurn(game.turn())} player is in checkmate`;
    } else if (game.in_draw()) {
        status = 'Game over, drawn position';
    } else {
        status = `${_longTurn(game.turn())} player's chance to move`;

        if (game.in_check()) {
            status += `, ${_longTurn(game.turn())} player is in check`;
        }
    }

    _fen = game.fen()
    _pgn = game.pgn()

    $('#msg').text(status)
    $('#fen').text(_fen.substring(_fen.length - 100, _fen.length))
    $('#pgn').text(_pgn.substring(_pgn.length - 100, _pgn.length))
};
/**
 * Feed the shot form the player turn, which is 'w' or 'b' and return the
 * long capitalize form of the name.
 *
 * @param t also know as short term of the turn.
 * @return the long capitalized version of the turn.
 */
const _longTurn = (t) => t === 'w' ? 'WHITE' : 'BLACK';
/**
 * Set the cursor to an appropriate state given via the parameter 'c'.
 *
 * @param c cursor state to be set.
 * @param d display state of the ajax loader.
 */
const _wait = (c, d) => {
    $('.progress').css('display', d !== undefined && d.trim() !== '' ? d : 'none');
    $('html,body').css('cursor', c);
};
/**
 * Check if the FEN string is available from the session or not
 * and return the available string if available and return 0, if
 * not.
 */
const _fen_ = () => {
    const fen = $('#fen_hidden').text();
    return fen !== undefined && fen !== '' ? fen : 0;
}
/**
 * JQuery "ready" method to start the client side process and listeners.
 */
$(() => {
    const configs = {
        draggable: true,
        dropOffBoard: 'trash',
        moveSpeed: 'slow',
        snapbackSpeed: 500,
        snapSpeed: 100,
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd
    };

    game = Chess();
    /*
     * Setup the 'position' if there's a FEN in the session and load
     * the game from the FEN as well.
     */
    const fen = _fen_()
    if (fen) {
        configs.position = fen;
        game.load(fen);
    }

    board = ChessBoard('board1', configs);
    /*
     * Start the 'board' if the FEN string is not available. Which means
     * this is new game.
     */
    if (!fen) board.start();
    /*
     * Update the status base on the initialized board config.
     */
    updateStats()
    /*
     * Connect to the WebSocket Broker.
     */
    connect();
});