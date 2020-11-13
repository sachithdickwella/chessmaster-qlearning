let stompClient;
/**
 * Connect and subscribe a WebSocket. in order to receive server responses from
 * the model.
 */
const connect = () => {
    const socket = new SockJS('/gs-guide-websocket');
    stompClient = Stomp.over(socket);
    stompClient.connect({}, frame => {
        console.debug("Connected:", frame)
        stompClient.subscribe(`/session/${document.cookie.match('SID=([A-F0-9]+)(?:;)?')[1]}/topic/next`, nextMove);
    });
}
/**
 * Make the move upon the server response from the model.
 *
 * @param next to give in the next move location on the board.
 */
const nextMove = (next) => {
    console.info(JSON.parse(next.body))
}
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
 * JQuery "ready" method to start the client side process and listeners.
 */
$(() => {
    const board = ChessBoard('board1', {
        draggable: true,
        dropOffBoard: 'trash'
    });
    board.start();
    /**
     * Connect to the WebSocket Broker.
     */
    connect();
});