package com.traviard.chessmaster.controller;

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
    @Value("${app.name}")
    private String appName;
    /**
     * Application home page title in {@link String}.
     */
    @Value("${app.home.title}")
    private String appHomeTitle;

    /**
     * Update the {@code JSESSIONID} cookie attributes to with custom details
     *
     * @return an instance of {@link CookieSerializer} to bind a bean and update
     * the default session cookie.
     */
    @Bean
    public CookieSerializer cookieSerializer() {
        DefaultCookieSerializer serializer = new DefaultCookieSerializer();
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
     * @param model Instance of {@link Model}.
     * @return the {@link String} of the page name.
     */
    @GetMapping(path = {"/", "index", "index.html"})
    public String index(@NotNull Model model) {
        model.addAttribute("title", appHomeTitle);
        return "index";
    }
}
