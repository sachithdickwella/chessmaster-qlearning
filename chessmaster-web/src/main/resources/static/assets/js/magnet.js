let board;
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
}
/**
 * Make the move upon the server response from the model.
 *
 * @param next to give in the next move location on the board.
 */
const nextMove = (next) => {
    console.info(JSON.parse(next.body));
    board.move(`${players.black.k0}-c6`);
}
/**
 * The event of new move by the user.
 */
const onDrop = (source, target, piece, newPos, oldPos, orientation) => shot();
/**
 * Shoot the chessboard HTML as a canvas and push it to the server end as an image.
 */
const shot = () => html2canvas(document.querySelector("#board1")).then(canvas => push(canvas));
/**
 * Send out the shot out canvas to the backend server as an image/png via AJAX.
 *
 * @param canvas of the shot out chessboard.
 */
const push = (canvas) => {
    canvas.toBlob((blob) => {
        const form = new FormData()
        form.append('file', blob, 'board-frame')
        $.ajax({
            url: '/movement/grab',
            method: 'POST',
            enctype: 'multipart/form-data',
            contentType: false,
            processData: false,
            cache: false,
            data: form,
            beforeSend: (_ /* jqXHR */) => {
                /*
                 * Show ajax loader before send the frame and show until
                 * receive a response with the next move.
                 */
            },
            error: (jqXHR) => {
                console.log(jqXHR)
            }
        })
    }, 'image/png');
}
/**
 * Chess pieces. p# is Pawn, k# is Knight, r# is Rook, b# is bishop
 * and King and Queen pieces respectively with the players colors.
 */
const players = {
    white: {
        p0: "a2", p1: "b2", p2: "c2", p3: "d2", p4: "e2", p5: "f2", p6: "g2", p7: "h2",
        k0: "b1", k1: "g1",
        r0: "a1", r1: "h1",
        b0: "c1", b1: "f1",
        king: "d1",
        queen: "e1"
    },
    black: {
        p0: "a7", p1: "b7", p2: "c7", p3: "d7", p4: "e7", p5: "f7", p6: "g7", p7: "h7",
        k0: "b8", k1: "g8",
        r0: "a8", r1: "h8",
        b0: "c8", b1: "f8",
        king: "d8",
        queen: "e8"
    }
};
/**
 * JQuery "ready" method to start the client side process and listeners.
 */
$(() => {
    board = ChessBoard('board1', {
        draggable: true,
        dropOffBoard: 'trash',
        moveSpeed: 'slow',
        snapbackSpeed: 500,
        snapSpeed: 100,
        onDrop: onDrop
    });
    board.start();
    /**
     * Connect to the WebSocket Broker.
     */
    connect();
});