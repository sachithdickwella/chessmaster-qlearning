package com.traviard.chessmaster.controller;

import com.traviard.chessmaster.util.RunnableMode;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.annotation.ScopedProxyMode;
import org.springframework.session.web.http.CookieSerializer;
import org.springframework.session.web.http.DefaultCookieSerializer;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.context.annotation.SessionScope;

import javax.servlet.http.HttpServletRequest;
import java.util.Optional;

import static com.traviard.chessmaster.util.SessionConstants.FEN;
import static org.apache.commons.lang3.StringUtils.EMPTY;

/**
 * @author Sachith Dickwella
 */
@PropertySource("classpath:app-config.properties")
@SessionScope(proxyMode = ScopedProxyMode.TARGET_CLASS)
@Controller
public class WebMvcController {

    /**
     * Application title in {@link String}.
     */
    @SuppressWarnings("unused")
    @Value("${app.name}")
    private String appName;
    /**
     * Application home page title in {@link String}.
     */
    @Value("${app.home.title}")
    private String appHomeTitle;
    /**
     * Application execution mode from properties.
     */
    @Value("#{T(com.traviard.chessmaster.util.RunnableMode).of('${app.execution.mode}')}")
    private RunnableMode runMode;

    /**
     * Update the {@code JSESSIONID} cookie attributes to with custom details
     *
     * @return an instance of {@link CookieSerializer} to bind a bean and update
     * the default session cookie.
     */
    @Bean
    public CookieSerializer cookieSerializer() {
        final var serializer = new DefaultCookieSerializer();
        serializer.setCookieName("JSESSIONID");
        serializer.setCookiePath("/");
        serializer.setUseSecureCookie(true);
        serializer.setUseHttpOnlyCookie(true);
        serializer.setDomainNamePattern("^.+?\\.(\\w+\\.[a-z]+)$");
        serializer.setSameSite("Strict");
        return serializer;
    }

    /**
     * Model mapping for the home/main page.
     *
     * @param model   Instance of {@link Model} set outbound model content.
     * @param request for the current request instance from the front-end.
     * @return the {@link String} of the page name.
     */
    @GetMapping(path = {"/", "index", "index.html"})
    public String index(@NotNull Model model, @NotNull HttpServletRequest request) {
        model.addAttribute("title", appHomeTitle);
        model.addAttribute("fen", Optional.ofNullable(request.getSession()
                .getAttribute(FEN.attribute()))
                .orElse(EMPTY));
        model.addAttribute("mode", runMode.mode());

        return "index";
    }
}
