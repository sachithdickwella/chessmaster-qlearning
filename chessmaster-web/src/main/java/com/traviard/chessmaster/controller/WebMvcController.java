package com.traviard.chessmaster.controller;

import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

/**
 * @author Sachith Dickwella
 */
@PropertySource("classpath:app-config.properties")
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
